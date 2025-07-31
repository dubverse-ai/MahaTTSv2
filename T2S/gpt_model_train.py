import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import linecache
import mmap
import pickle as pkl
import random
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchaudio
import transformers
from accelerate import Accelerator, DistributedDataParallelKwargs
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm

import wandb
from config import config
from T2S.autoregressive import TS_model
from T2S.mel_spec import get_mel_spectrogram
from T2S.utilities import get_mask_from_lengths
from Text import code_labels, labels, text_labels

torch.manual_seed(config.seed_value)
np.random.seed(config.seed_value)
random.seed(config.seed_value)

# code encdec
text_enc = {j: i for i, j in enumerate(text_labels)}
text_dec = {i: j for i, j in enumerate(text_labels)}

# text encdec
code_enc = {j: i for i, j in enumerate(code_labels)}
code_dec = {i: j for i, j in enumerate(code_labels)}


def read_specific_line(filename, line_number):
    line = linecache.getline(filename, line_number)
    return line.strip()  # Remove any leading or trailing whitespace


CLIP_LENGTH = config.CLIP_LENGTH


class semantic_dataset(Dataset):
    def __init__(
        self,
        transcript_path,
        semantic_path=None,
        ref_mels_path=None,
        ref_k=1,
        scale=True,
    ):
        super().__init__()
        self.scale = scale
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

            self.data = [[i, semb[i], data[i]] for i in data.keys()]

        else:
            line_index = {}
            with open(transcript_path, "rb") as file:
                mmapped_file = mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ)
                line_number = 0
                offset = 0
                pbar = tqdm()
                while offset < len(mmapped_file):
                    line_index[line_number] = offset
                    offset = mmapped_file.find(b"\n", offset) + 1
                    line_number += 1
                    pbar.update(1)
                pbar.close()
                self.mmapped_file = mmapped_file
            self.line_index = line_index
            self.data_len = len(line_index)
            print("data length:", self.data_len)
            self.transcript_path = transcript_path

        self.ref_k = ref_k
        self.max_wav_value = config.MAX_WAV_VALUE

    def get_mel(self, filepath):
        audio_norm, sampling_rate = torchaudio.load(filepath)
        melspec = get_mel_spectrogram(audio_norm, sampling_rate).squeeze(0)
        energy = []
        return melspec, list(energy)

    def __len__(self):
        if self.scale:
            return self.data_len
        return len(self.data)

    def __getitem__(self, index) -> Any:
        if not self.scale:
            lang, path, semb, text = self.data[index]
            ref_mels = self.ref_mels[path][: self.ref_k]

        else:
            self.mmapped_file.seek(self.line_index[index])
            line = self.mmapped_file.readline().decode("utf-8")

            try:
                lang, path, text, semb_ids = line.split("|")
            except Exception as e:
                print(index, line)
                if index + 1 < self.data_len:
                    return self.__getitem__(index + 1)
                return self.__getitem__(0)
            semb = semb_ids.split()
            ref_mels = [path]
            # ref_mels = [i.split(',') for i in ref_mels.split('\t')][:self.ref_k]

        if len(semb) < 5:
            print(index, "No Semb tokens found")
            if index + 1 < self.data_len:
                return self.__getitem__(index + 1)
            return self.__getitem__(0)

        if len(ref_mels) == 0:
            ref_mels.append((path, 1))
            ref_mels.append((path, 1))
            ref_mels.append((path, 1))

        while len(ref_mels) < self.ref_k:
            ref_mels.append(ref_mels[-1])

        text = text.lower().strip()
        try:
            text_ids = (
                [text_enc["<S>"]] + [text_enc[i] for i in text] + [text_enc["<E>"]]
            )
            semb_ids = (
                [code_enc["<SST>"]] + [code_enc[i] for i in semb] + [code_enc["<EST>"]]
            )
        except Exception as e:
            print(index, e)
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
            ref_mels = [self.get_mel(path)[0] for path in ref_mels]
        except Exception as e:
            print(index, e, path)
            if index + 1 < self.data_len:
                return self.__getitem__(index + 1)
            return self.__getitem__(0)

        ref_c = []
        for i in range(self.ref_k):
            if ref_mels[i] is None:
                continue
            ref_c.append(ref_mels[i])

        if len(ref_c) == 0:
            if index + 1 < self.data_len:
                return self.__getitem__(index + 1)
            return self.__getitem__(0)

        if len(ref_c) != self.ref_k:
            while len(ref_c) < self.ref_k:
                ref_c.append(ref_c[-1])

        ref_mels = ref_c

        max_target_len = max([x.size(1) for x in ref_mels])
        ref_mels_padded = (
            torch.randn((self.ref_k, config.n_mel_channels, max_target_len))
        ) * 1e-9
        mel_length = []
        for i, mel in enumerate(ref_mels):
            ref_mels_padded[i, :, : mel.size(1)] = mel
            mel_length.append(mel.shape[-1])

        ref_mels = get_random_portion(ref_mels_padded, torch.tensor(mel_length))

        return {
            "text_ids": text_ids,
            "semb_ids": semb_ids,
            "ref_mels": ref_mels,
            "lang": torch.tensor(config.lang_index[lang]),
        }


def get_padded_seq(sequences, pad_random, before=False, pad__=0):
    max_len = max([len(s) for s in sequences])
    seq_len = []
    for i in range(len(sequences)):
        seq_len.append(len(sequences[i]))
        if pad_random:
            pad_ = pad_ = list((np.random.rand(max_len - len(sequences[i]))) * 1e-9)
        else:
            pad_ = [pad__] * (max_len - len(sequences[i]))
        if not before:
            sequences[i] = sequences[i] + pad_
        else:
            sequences[i] = pad_ + sequences[i]

    return sequences, seq_len


def collate(batch):
    text_ids = []
    semb_ids = []
    ref_mels = []
    langs = []

    for b in batch:
        text_ids.append(b["text_ids"])
        semb_ids.append(b["semb_ids"])
        ref_mels.append(b["ref_mels"])
        langs.append(b["lang"])

    text_ids, text_len = get_padded_seq(
        text_ids, pad_random=False, before=False, pad__=text_enc["<E>"]
    )
    code, code_len = get_padded_seq(semb_ids, pad_random=False, pad__=code_enc["<EST>"])

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
    ) * 1e-9

    for i, mel in enumerate(ref_mels):
        ref_mels_padded[i, :, :, : mel.size(-1)] = mel

    return (
        torch.tensor(text_ids),
        torch.tensor(code),
        torch.tensor(text_len),
        torch.tensor(code_len),
        ref_mels_padded,
        torch.tensor(langs),
    )


def train(model, train_dataset, val_dataset, save_dir, checkpoint_initial=None):
    accelerator = Accelerator(
        gradient_accumulation_steps=config.ts_gradient_accumulation_steps
    )  # ,kwargs_handlers=[ddp_kwargs]) mixed_precision="fp16",

    if config.ts_wandb_logs and accelerator.is_local_main_process:
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

    optimizer = optim.Adam(
        model.parameters(), lr=config.ts_lr, weight_decay=config.ts_weight_decay
    )
    # optimizer = transformers.Adafactor(model.parameters(), lr=config.ts_lr,weight_decay=config.ts_weight_decay, relative_step =False, scale_parameter =False)
    lr = config.ts_lr
    step_num = 0
    start_epoch = 0
    if checkpoint_initial is not None:
        model.load_state_dict(
            torch.load(checkpoint_initial, map_location=torch.device("cpu"))["model"],
            strict=True,
        )
        if (
            config.ts_finetuning
        ):  # freezing heads results in less hallucinations after Ft.
            for param in model.text_head.parameters():
                param.requires_grad = False

            for param in model.code_head.parameters():
                param.requires_grad = False

        model.train()

        print("loading optimizer")
        optimizer.load_state_dict(
            torch.load(checkpoint_initial, map_location=torch.device("cpu"))[
                "optimizer"
            ]
        )
        step_num = (
            int(
                torch.load(checkpoint_initial, map_location=torch.device("cpu"))["step"]
            )
            + 1
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
        print(f"Resuming training from epoch {start_epoch} and step {step_num}")

    train_dataloader, val_dataloader, model, optimizer = accelerator.prepare(
        train_dataset, val_dataset, model, optimizer
    )
    val_dataloader = val_dataset
    min_val_loss = 1000
    model.train()

    for i in range(start_epoch, config.ts_epochs):
        epoch_loss = []
        if accelerator.is_main_process:
            train_loader = tqdm(
                train_dataloader,
                desc="Rank %d: Training epoch %d"
                % (accelerator.local_process_index, i),
            )
        else:
            train_loader = train_dataloader

        for n, inputs in enumerate(train_loader):
            with accelerator.accumulate(model):
                # with accelerator.autocast():
                text_ids, code, text_len, code_len, ref_clips, langs = inputs
                mask_text = get_mask_from_lengths(text_len)
                code_mask = get_mask_from_lengths(code_len)
                attn_mask = torch.cat([mask_text, code_mask], dim=1)
                loss_text, loss_code, _ = model(
                    text_ids=text_ids,
                    ref_clips=ref_clips,
                    codes_ids=code,
                    language=langs,
                    return_loss=True,
                    attn_mask=attn_mask,
                )

                loss_text *= mask_text[:, 1:].float()
                loss_text = loss_text.sum() / mask_text[:, 1:].sum()

                loss_code *= code_mask[:, 1:].float()
                loss_code = loss_code.sum() / code_mask[:, 1:].sum()

                loss = loss_text * config.text_loss_weight + loss_code

                accelerator.backward(loss)
                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                step_num += 1

                if (
                    step_num % config.ts_gradient_accumulation_steps == 0
                    and config.ts_wandb_logs
                    and accelerator.is_main_process
                ):
                    wandb_log.log(
                        {
                            "training_loss": loss.item(),
                            "step": step_num // config.ts_gradient_accumulation_steps,
                        }
                    )

            epoch_loss.append(loss.item())

            if (
                not config.ts_finetuning
                and step_num
                % (config.ts_gradient_accumulation_steps * config.ts_eval_step)
                == 0
            ):
                val_loss = val(model, val_dataloader, accelerator.is_main_process)
                val_loss = accelerator.gather_for_metrics(val_loss).mean().item()
                model.train()
                if config.ts_wandb_logs and accelerator.is_main_process:
                    wandb_log.log(
                        {
                            "val_loss": val_loss,
                            "epoch": i,
                            "scheduled_learning_rate": lr,
                            "step": step_num // config.ts_gradient_accumulation_steps,
                        }
                    )

                if val_loss < min_val_loss:
                    # save the model
                    accelerator.wait_for_everyone()
                    unwrapped_model = accelerator.unwrap_model(model)
                    checkpoint = {
                        "epoch": i,
                        "step": str(step_num // config.gradient_accumulation_steps),
                        "model": unwrapped_model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                    }
                    torch.save(
                        checkpoint,
                        os.path.join(config.save_root_dir, "_best.pt"),
                    )
                    min_val_loss = val_loss

                # save the latest checkpoint
                accelerator.wait_for_everyone()
                unwrapped_model = accelerator.unwrap_model(model)
                checkpoint = {
                    "epoch": i,
                    "step": str(step_num // config.gradient_accumulation_steps),
                    "model": unwrapped_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                }
                torch.save(
                    checkpoint,
                    os.path.join(config.save_root_dir, str(step_num // config.gradient_accumulation_steps) + "_latest.pt"),
                )
                print(f"Saved latest checkpoint at {os.path.join(config.save_root_dir, str(step_num // config.gradient_accumulation_steps) + '_latest.pt')}")

        val_loss = val(model, val_dataloader, accelerator.is_main_process)
        val_loss = accelerator.gather_for_metrics(val_loss).mean().item()
        model.train()
        if config.ts_wandb_logs and accelerator.is_main_process:
            wandb_log.log(
                {
                    "val_loss": val_loss,
                    "epoch": i,
                    "scheduled_learning_rate": lr,
                    "step": step_num // config.ts_gradient_accumulation_steps,
                }
            )

        if val_loss < min_val_loss:
            # save the model
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            checkpoint = {
                "epoch": i,
                "step": str(step_num // config.gradient_accumulation_steps),
                "model": unwrapped_model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            torch.save(
                checkpoint, os.path.join(config.save_root_dir, "best.pt")
            )
            min_val_loss = val_loss
        print(f"Saved best checkpoint at {os.path.join(config.save_root_dir, 'best.pt')}")
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        checkpoint = {
            "epoch": i,
            "step": str(step_num // config.gradient_accumulation_steps),
            "model": unwrapped_model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        torch.save(
            checkpoint,
            os.path.join(config.save_root_dir, str(i) + "_latest.pt"),
        )
        
        if config.ts_wandb_logs and accelerator.is_local_main_process:
            wandb_log.log(
                {
                    "scheduled_learning_rate": lr,
                    "epoch": i,
                    "step": step_num // config.ts_gradient_accumulation_steps,
                }
            )
        print(
            "epoch_number : ", i, " training loss : ", sum(epoch_loss) / len(epoch_loss)
        )

    if config.ts_wandb_logs and accelerator.is_local_main_process:
        wandb_log.finish()


def val(model, val_dataloader, _main=False):
    """
    Return the loss value
    """
    print("VALIDATION STARTING:")
    model.eval()
    val_loss = []
    device = next(model.parameters()).device
    if _main:
        val_dataloader = tqdm(val_dataloader)
    with torch.no_grad():
        for inputs in val_dataloader:
            text_ids, code, text_len, code_len, ref_clips, langs = inputs
            mask_text = get_mask_from_lengths(text_len).to(device)
            code_mask = get_mask_from_lengths(code_len).to(device)
            attn_mask = torch.cat([mask_text, code_mask], dim=1)
            loss_text, loss_code, _ = model(
                text_ids=text_ids.to(device),
                ref_clips=ref_clips.to(device),
                codes_ids=code.to(device),
                language=langs.to(device),
                return_loss=True,
                attn_mask=attn_mask,
            )

            loss_text *= mask_text[:, 1:].float()
            loss_text = loss_text.sum() / mask_text[:, 1:].sum()
            loss_code *= code_mask[:, 1:].float()
            loss_code = loss_code.sum() / code_mask[:, 1:].sum()
            loss = loss_text * config.text_loss_weight + loss_code

            val_loss.append(loss.item())

    val_loss = sum(val_loss) / len(val_loss)
    print(" Validation loss : ", val_loss)
    return torch.tensor(val_loss).to(device)


def main():

    os.makedirs(os.path.join(config.save_root_dir, config.model_name, "T2S"), exist_ok=True)

    file_name_train = config.train_file
    file_name_val = config.val_file

    checkpoint = config.t2s_checkpoint
    model = TS_model(n_embed=1024, n_layer=30, n_head=16)

    val_dataset = DataLoader(
        semantic_dataset(file_name_val, scale=True),
        pin_memory=True,
        persistent_workers=True,
        num_workers=2,
        batch_size=config.ts_batch_size,
        shuffle=True,
        drop_last=False,
        collate_fn=collate,
    )

    train_dataset_ = semantic_dataset(file_name_train, scale=True)
    train_dataset = DataLoader(
        train_dataset_,
        pin_memory=True,
        persistent_workers=True,
        num_workers=config.ts_num_workers,
        batch_size=config.ts_batch_size,
        shuffle=True,
        drop_last=False,
        collate_fn=collate,
    )
    
    train(
        model,
        train_dataset,
        val_dataset,
        save_dir=os.path.join(config.save_root_dir, config.model_name, "T2S"),
        checkpoint_initial=checkpoint
    )


if __name__ == "__main__":
    main()
