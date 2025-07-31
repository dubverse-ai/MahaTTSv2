import os
import sys
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import linecache
import mmap
import pickle as pkl
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchaudio
from accelerate import Accelerator, DistributedDataParallelKwargs
from mel_spec import get_mel_spectrogram
from torch.distributed import destroy_process_group, init_process_group
from torch.distributed.elastic.utils.data import ElasticDistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm

import wandb
from config import config
from S2A.diff_model import DiffModel
from S2A.flow_matching import BASECFM
from S2A.inference import infer
from S2A.utilities import (dynamic_range_compression, get_mask,
                           get_mask_from_lengths, load_wav_to_torch,
                           normalize_tacotron_mel)
from Text import code_labels, labels, text_labels

# import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

torch.manual_seed(config.seed_value)
np.random.seed(config.seed_value)
random.seed(config.seed_value)

CLIP_LENGTH = config.CLIP_LENGTH

# code encdec
text_enc = {j: i for i, j in enumerate(text_labels)}
text_dec = {i: j for i, j in enumerate(text_labels)}

# text encdec
code_enc = {j: i for i, j in enumerate(code_labels)}
code_dec = {i: j for i, j in enumerate(code_labels)}


def read_specific_line(filename, line_number):
    line = linecache.getline(filename, line_number)
    return line.strip()


class Acoustic_dataset(Dataset):
    def __init__(
        self,
        transcript_path,
        semantic_path=None,
        ref_mels_path=None,
        ref_k=1,
        scale=True,
        ar_active=False,
        clip=True,
        dur_=None,
    ):
        super(Acoustic_dataset).__init__()
        self.scale = scale
        self.ar_active = ar_active
        self.clip = clip
        self.dur_ = dur_
        if self.dur_ is None:
            self.dur_ = 2
        if not scale:
            with open(transcript_path, "r") as file:
                data = file.read().strip("\n").split("\n")[:]

            with open(semantic_path, "r") as file:
                semb = file.read().strip("\n").split("\n")

            with open(ref_mels_path, "rb") as file:
                self.ref_mels = pkl.load(file)

            semb = {
                i.split("\t")[0]: [j for j in i.split("\t")[1].split()] for i in semb
            }
            data = {i.split("|")[0]: i.split("|")[1].strip().lower() for i in data}

            self.data = [[i, semb[i], data[i]] for i in data.keys()][:]

        else:
            self.transcript_path = transcript_path
            line_index = {}
            with open(transcript_path, "rb") as file:
                mmapped_file = mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ)
                line_number = 0
                offset = 0
                progress_bar = tqdm(desc="processing:")
                while offset < len(mmapped_file):
                    line_index[line_number] = offset
                    offset = mmapped_file.find(b"\n", offset) + 1
                    line_number += 1
                    progress_bar.update(1)
                progress_bar.close()
                self.mmapped_file = mmapped_file
            self.data_len = len(line_index)
            self.line_index = line_index

        self.ref_k = ref_k
        self.max_wav_value = config.MAX_WAV_VALUE

    def get_mel(self, filepath, semb_ids=None, align=False, ref_clip=False):
        audio_norm, sampling_rate = torchaudio.load(filepath)
        dur = audio_norm.shape[-1] / sampling_rate

        if self.clip and dur > self.dur_ and align:
            max_audio_start = int(dur - self.dur_)
            if max_audio_start <= 0:
                audio_start = 0
            else:
                audio_start = np.random.randint(0, max_audio_start)

            audio_norm = audio_norm[
                :,
                audio_start * sampling_rate : (audio_start + self.dur_) * sampling_rate,
            ]
            semb_ids = semb_ids[audio_start * 50 : ((audio_start + self.dur_) * 50) - 1]

        #     86 mel -> 1s for 22050 setting
        #     93 mel -> 1s for 24000 setting

        if ref_clip == True:
            dur_ = 6
            max_audio_start = int(dur - dur_)
            if max_audio_start <= 0:
                audio_start = 0
            else:
                audio_start = np.random.randint(0, max_audio_start)
            audio_norm = audio_norm[
                :, audio_start * sampling_rate : (audio_start + dur_) * sampling_rate
            ]

        melspec = get_mel_spectrogram(audio_norm, sampling_rate).squeeze(0)
        energy = []
        if align:
            return melspec, list(energy), semb_ids
        return melspec, list(energy)

    def __len__(self):
        if self.scale:
            return self.data_len
        return len(self.data)

    def __getitem__(self, index):
        """
        mel_spec,semb
        """
        if not self.scale:
            lang, path, semb, text = self.data[index]
            ref_mels = self.ref_mels[path][: self.ref_k]
            semb_ids = [int(i) + 1 for i in semb]  # 0 for pad

        else:
            self.mmapped_file.seek(self.line_index[index])
            line = self.mmapped_file.readline().decode("utf-8")

            lang, path, text, semb_ids = line.split("|")
            semb_ids = [int(i) + 1 for i in semb_ids.split()]
            ref_mels = [path][: self.ref_k]

        try:
            mel_spec, energy, semb_ids = self.get_mel(path, semb_ids, align=True)
            if len(semb_ids) == 0:
                raise Exception("Sorry, no semb ids" + str(line))
        except Exception as e:
            print(index, e)
            if index + 1 < self.data_len:
                return self.__getitem__(index + 1)
            return self.__getitem__(0)

        if len(ref_mels) == 0:
            print(index, e, "no ref mels present")
            if index + 1 < self.data_len:
                return self.__getitem__(index + 1)
            return self.__getitem__(0)

        while len(ref_mels) < self.ref_k:
            ref_mels.append(ref_mels[-1])

        if mel_spec is None:
            print(index, e, "mel_spec error present")
            if index + 1 < self.data_len:
                return self.__getitem__(index + 1)
            return self.__getitem__(0)

        def get_random_portion(mel, mask_lengths):
            clip = mask_lengths <= CLIP_LENGTH
            ref_mel = mel[:, :, :CLIP_LENGTH].clone()
            for n, z in enumerate(clip):
                if not z:
                    start = np.random.randint(0, mask_lengths[n].item() - CLIP_LENGTH)
                    ref_mel[n, :, :] = mel[n, :, start : start + CLIP_LENGTH].clone()
            return ref_mel

        try:
            ref_mels = [self.get_mel(path, ref_clip=True)[0] for path in ref_mels]
        except Exception as e:
            print(index, e, "ref_mels mel_spec error")
            if index + 1 < self.data_len:
                return self.__getitem__(index + 1)
            return self.__getitem__(0)

        ref_c = []
        for i in range(self.ref_k):
            if ref_mels[i] is None:
                continue
            ref_c.append(ref_mels[i])

        if len(ref_c) == 0:
            print("no refs mel spec found")
            if index + 1 < self.data_len:
                return self.__getitem__(index + 1)
            return self.__getitem__(0)

        if len(ref_c) != self.ref_k:
            while len(ref_c) < self.ref_k:
                ref_c.append(ref_c[-1])

        ref_mels = ref_c
        max_target_len = max([x.size(1) for x in ref_mels])
        ref_mels_padded = (
            torch.randn((self.ref_k, config.n_mel_channels, max_target_len)) * 1e-9
        )
        mel_length = []
        for i, mel in enumerate(ref_mels):
            ref_mels_padded[i, :, : mel.size(1)] = mel
            mel_length.append(mel.shape[-1])

        ref_mels = get_random_portion(ref_mels_padded, torch.tensor(mel_length))

        text_ids = (
            [text_enc["<S>"]]
            + [text_enc[i] for i in text.strip() if i in text_enc]
            + [text_enc["<E>"]]
        )
        if self.ar_active:
            semb_ids = (
                [code_enc["<SST>"]]
                + [code_enc[str(i - 1)] for i in semb_ids]
                + [code_enc["<EST>"]]
            )

        return {
            "mel": mel_spec,
            "code": semb_ids,
            "path": path,
            "ref_mels": ref_mels,
            "text": text_ids,
        }  # , 'ref_mel_length':mel_length}


def get_padded_seq(sequences, pad_random, before=False, pad__=0):
    max_len = max([len(s) for s in sequences])
    seq_len = []
    for i in range(len(sequences)):
        seq_len.append(len(sequences[i]))
        if pad_random:
            pad_ = list((np.random.rand(max_len - len(sequences[i]))) * 1e-9)
        else:
            pad_ = [pad__] * (max_len - len(sequences[i]))
        if not before:
            sequences[i] = sequences[i] + pad_
        else:
            sequences[i] = pad_ + sequences[i]

    return sequences, seq_len


def collate(batch):
    mel_specs = []
    code = []
    paths = []
    ref_mels = []
    text_ids = []

    for b in batch:
        mel_specs.append(b["mel"])
        code.append(b["code"])
        paths.append(b["path"])
        ref_mels.append(b["ref_mels"])
        text_ids.append(b["text"])

    if code[-1][-1] == code_enc["<EST>"]:
        code, code_len = get_padded_seq(code, pad_random=False, pad__=code_enc["<PAD>"])
    else:
        code, code_len = get_padded_seq(code, pad_random=False)

    text_ids, text_len = get_padded_seq(
        text_ids, pad_random=False, before=True, pad__=text_enc["<PAD>"]
    )
    ref_max_target_len = max([x.size(-1) for x in ref_mels])
    ref_mels_padded = (
        torch.randn(
            (
                len(batch),
                ref_mels[0].shape[0],
                config.n_mel_channels,
                ref_max_target_len,
            )
        )
        * 1e-9
    )

    for i, mel in enumerate(ref_mels):
        ref_mels_padded[i, :, :, : mel.size(-1)] = mel

    max_target_len = max([x.size(-1) for x in mel_specs])
    mel_padded = torch.randn((len(batch), config.n_mel_channels, max_target_len)) * 1e-9
    mel_length = []
    for i, mel in enumerate(mel_specs):
        mel_padded[i, :, : mel.size(-1)] = mel
        mel_length.append(mel.shape[-1])

    return (
        normalize_tacotron_mel(mel_padded),
        torch.tensor(code),
        torch.tensor(mel_length),
        torch.tensor(code_len),
        ref_mels_padded,
        torch.tensor(text_ids),
        torch.tensor(text_len),
        paths,
    )


def train(
    model,
    diffuser,
    train_dataloader,
    val_dataloader,
    schedule_sampler=None,
    rank=0,
    ar_active=False,
    m1=None,
    checkpoint_initial=None,
):
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        kwargs_handlers=[ddp_kwargs],
    )
    if config.sa_wandb_logs and accelerator.is_local_main_process:
        conf_ = {}
        for i, j in config.__dict__.items():
            conf_[str(i)] = str(j)
        wandb_log = wandb.init(
            project=config.wandb_project,
            entity=config.user_name,
            name=config.model_name,
            config=conf_,
        )
        wandb_log.watch(model, log_freq=100)
    else:
        wandb_log = None

    model.train()
    optimizer = optim.AdamW(
        model.parameters(), lr=config.sa_lr, weight_decay=config.sa_weight_decay
    )
    lr = config.sa_lr
    min_val_loss = 1000
    step_num = 0
    start_epoch = 0
    if checkpoint_initial is not None:
        print(checkpoint_initial)
        model.load_state_dict(
            torch.load(checkpoint_initial, map_location=torch.device("cpu"))["model"],
            strict=True,
        )
        model.train()
        optimizer.load_state_dict(
            torch.load(checkpoint_initial, map_location=torch.device("cpu"))[
                "optimizer"
            ]
        )
        step_num = int(
            torch.load(checkpoint_initial, map_location=torch.device("cpu"))["step"]
        )
        step_num = 0
        start_epoch = (
            int(
                torch.load(checkpoint_initial, map_location=torch.device("cpu"))[
                    "epoch"
                ]
            )
            + 1
        )
        print(f"resuming training from epoch {start_epoch} and step {step_num}")

    train_dataloader, model, optimizer = accelerator.prepare(
        train_dataloader, model, optimizer
    )

    FM = BASECFM()
    device = next(model.parameters()).device
    if ar_active:
        m1 = m1.to(device)

    loading_time = []
    for i in range(start_epoch, config.sa_epochs):
        epoch_loss = {"vlb": [], "mse": [], "loss": []}
        if accelerator.is_local_main_process:
            train_loader = tqdm(train_dataloader, desc="Training epoch %d" % (i))
        else:
            train_loader = train_dataloader

        for inputs in train_loader:
            with accelerator.accumulate(model):
                optimizer.zero_grad()
                x1, code_emb, mask_lengths, _, ref_mels, text_ids, _, _ = inputs
                mask = get_mask_from_lengths(mask_lengths).unsqueeze(1)
                mask = mask.squeeze(1).float()

                loss, _, t = FM.compute_loss(model, x1, mask, code_emb, ref_mels)

                accelerator.backward(loss)
                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                step_num += 1

                epoch_loss["loss"].append(loss.item())

                if step_num % config.gradient_accumulation_steps == 0:
                    epoch_training_loss = torch.tensor(
                        sum(epoch_loss["loss"]) / len(epoch_loss["loss"])
                    ).to(device)
                    epoch_loss = {"vlb": [], "mse": [], "loss": []}
                    epoch_training_loss = (
                        accelerator.gather_for_metrics(epoch_training_loss)
                        .mean()
                        .item()
                    )

                    if config.sa_wandb_logs and accelerator.is_local_main_process:
                        wandb_log.log({"training_loss": epoch_training_loss})

            if (
                step_num % (config.sa_eval_step * config.gradient_accumulation_steps)
                == 0
            ):
                print(f"evaluation at step_num {step_num}")
                if accelerator.is_local_main_process:
                    # save the latest checkpoint
                    unwrapped_model = accelerator.unwrap_model(model)
                    checkpoint = {
                        "epoch": i,
                        "step": step_num // config.gradient_accumulation_steps,
                        "model": unwrapped_model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "norms": config.norms,
                    }
                    torch.save(
                        checkpoint,
                        os.path.join(config.save_root_dir, "latest.pt",),
                    )

                if accelerator.is_local_main_process:
                    val_loss, val_mse, val_vlb, time_steps_mean = val(
                        model,
                        FM,
                        val_dataloader,
                        infer_=config.sa_infer,
                        epoch=i,
                        rank=accelerator.is_local_main_process,
                        ar_active=ar_active,
                        m1=m1,
                    )
                    model.train()

                    print(
                        "validation loss : ",
                        val_loss,
                        "\nvalidation mse loss : ",
                        val_mse,
                        "\nvalidation vlb loss : ",
                        val_vlb,
                    )
                    if config.sa_wandb_logs:
                        wandb_log.log({"val_loss": val_loss})
                    if val_loss < min_val_loss:
                        unwrapped_model = accelerator.unwrap_model(model)
                        checkpoint = {
                            "epoch": i,
                            "step": step_num // config.gradient_accumulation_steps,
                            "model": unwrapped_model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "norms": config.norms,
                        }
                        torch.save(
                            checkpoint,
                            os.path.join(config.save_root_dir, "_best.pt"),
                        )
                        min_val_loss = val_loss

        if i == start_epoch + 12:
            exit()
    if config.sa_wandb_logs and accelerator.is_local_main_process:
        wandb_log.finish()


def val(
    model,
    FM,
    val_dataloader,
    infer_=False,
    epoch=0,
    rank=False,
    ar_active=False,
    m1=None,
):
    """
    Return the loss value
    """
    model.eval()
    epoch_loss = {"vlb": [], "mse": [], "loss": []}
    code_emb = None
    x = None
    mask_lengths = None
    time_steps_mean = []
    device = next(model.parameters()).device
    if rank:
        val_dataloader = tqdm(val_dataloader, desc="validation epoch %d" % (epoch))
    else:
        val_dataloader = val_dataloader

    with torch.no_grad():
        for inputs in val_dataloader:
            x1, code_emb, mask_lengths, code_len, ref_mels, text_ids, _, _ = inputs

            mask = get_mask_from_lengths(mask_lengths).unsqueeze(1).to(device)
            mask = mask.squeeze(1).float()
            x1 = x1.to(device)
            code_emb = code_emb.to(device)
            text_ids = text_ids.to(device)
            ref_mels = ref_mels.to(device)

            loss, _, t = FM.compute_loss(model, x1, mask, code_emb, ref_mels)
            time_steps_mean.extend(t.detach().cpu().squeeze(-1).squeeze(-1).tolist())
            mse = loss
            vlb = loss

            epoch_loss["loss"].append(loss.item())
            epoch_loss["mse"].append(mse.item())
            epoch_loss["vlb"].append(vlb.item())

    epoch_vlb_loss = sum(epoch_loss["vlb"]) / len(epoch_loss["vlb"])
    epoch_training_loss = sum(epoch_loss["loss"]) / len(epoch_loss["loss"])
    epoch_mse_loss = sum(epoch_loss["mse"]) / len(epoch_loss["mse"])
    if rank and infer_ and epoch % config.sa_infer_epoch == 0:
        k = 4
        if ar_active:
            code_embs = [code_emb[i, :, : code_len[i]] for i in range(k)]
        else:
            code_embs = [code_emb[i, : code_len[i]] for i in range(k)]
        audio_paths, mels = infer(
            model, mask_lengths[:k], code_embs, ref_mels[:k, :], epoch
        )

        if config.sa_wandb_logs:
            images = [
                wandb.Image(mel[0], caption="epoch: " + str(epoch)) for mel in mels
            ]
            x = [
                wandb.Image(x1[i, :, : mask_lengths[i]], caption="Actual: ")
                for i in range(k)
            ]
            wandb.log(
                {
                    "predicted audio": [
                        wandb.Audio(audio_path) for audio_path in audio_paths
                    ],
                    "predicted melspec": images,
                    "actual melspec": x,
                    "epoch": epoch,
                }
            )

    return (
        epoch_training_loss,
        epoch_mse_loss,
        epoch_vlb_loss,
        sum(time_steps_mean) / len(time_steps_mean),
    )


if __name__ == "__main__":
    os.makedirs(os.path.join(config.save_root_dir, config.model_name, "S2A"), exist_ok=True)
    
    model = DiffModel(
        input_channels=100,
        output_channels=100,
        model_channels=512,  # 1024
        num_heads=8,  # 16
        dropout=0.10,
        num_layers=8,
        enable_fp16=False,
        condition_free_per=0.0,
        multispeaker=True,
        style_tokens=100,
        training=True,
        ar_active=False,
        in_latent_channels=len(code_labels),
    )
    m1 = None
    checkpoint = None
    print("Model Loaded")
    print("batch_size :", config.sa_batch_size)
    print("Diffusion timesteps:", config.sa_timesteps_max)

    file_name_train = config.train_file
    file_name_val = config.val_file

    train_dataset = Acoustic_dataset(file_name_train, scale=config.scale)
    train_dataloader = DataLoader(
        train_dataset,
        pin_memory=True,
        persistent_workers=True,
        num_workers=config.sa_num_workers,
        batch_size=config.sa_batch_size,
        shuffle=True,
        drop_last=False,
        collate_fn=collate,
    )

    val_dataset = Acoustic_dataset(file_name_val, scale=config.scale, dur_=5)
    val_dataloader = DataLoader(
        val_dataset,
        pin_memory=True,
        persistent_workers=True,
        num_workers=config.sa_num_workers,
        batch_size=config.sa_batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=collate,
    )

    train(
        model,
        diffuser=None,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        rank=0,
        ar_active=False,
        m1=m1,
        checkpoint_initial=checkpoint,
    )
