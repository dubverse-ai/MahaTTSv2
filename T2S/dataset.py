import os
import sys
from typing import Any

sys.path.append("../")
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
import transformers
from accelerate import Accelerator, DistributedDataParallelKwargs
from autoregressive import TS_model
from cleaners import english_cleaners
from librosa.filters import mel as librosa_mel_fn
from mel_spec import get_mel_spectrogram
from meta_stats import process_file, process_file_for_heads
from stft import STFT
from torch.utils.data import (DataLoader, Dataset, WeightedRandomSampler,
                              get_worker_info)
from tqdm import tqdm
from utilities import get_mask_from_lengths

import wandb
from config import config
from Text import code_labels, labels, text_labels

torch.manual_seed(config.seed_value)
np.random.seed(config.seed_value)
random.seed(config.seed_value)
print(text_labels)
# add semantic tokens:
# tok_enc = {j:i for i,j in enumerate(labels)}
# tok_dec = {j:i for i,j in enumerate(labels)}

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


class semantic_dataset_batch(Dataset):
    def __init__(
        self,
        transcript_path,
        semantic_path=None,
        ref_mels_path=None,
        ref_k=3,
        scale=False,
        process_id=None,
        total_processes=None,
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
            # with open(transcript_path,'r') as file:
            # get meta for dataset
            # for count, line in enumerate(file):
            #     pass
            # count = 80
            print(transcript_path)
            # self.weights,self.count = process_file(transcript_path)
            self.heads, self.weights, self.count = process_file_for_heads(
                transcript_path, total_processes, process_id
            )
            print("length :", self.count)
            self.data_len = self.count
            self.transcript_path = transcript_path
            line_index = {}
            with open(transcript_path, "rb") as file:
                mmapped_file = mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ)
                line_number = 0
                offset = 0
                while offset < len(mmapped_file):
                    line_index[line_number] = offset
                    offset = mmapped_file.find(b"\n", offset) + 1
                    # print(line_number,offset)
                    line_number += 1
                self.mmapped_file = mmapped_file
            self.line_index = line_index

            self.process_id = process_id
            self.total_processes = total_processes
            self.iterator = None

        self.ref_k = ref_k
        self.max_wav_value = config.MAX_WAV_VALUE
        self.stft_fn = STFT(config.filter_length, config.hop_length, config.win_length)

        mel_basis = librosa_mel_fn(
            sr=config.sampling_rate,
            n_fft=config.filter_length,
            n_mels=config.n_mel_channels,
            fmin=config.mel_fmin,
            fmax=config.mel_fmax,
        )

        self.mel_basis = torch.from_numpy(mel_basis).float()

    def get_mel(self, filepath):
        # audio, sampling_rate = load_wav_to_torch(filepath)
        # audio_norm = audio / self.max_wav_value
        audio_norm, sampling_rate = torchaudio.load(filepath)

        # dur = audio_norm.shape[-1]/sampling_rate

        # if dur<0.5:
        #     return None,None,None

        # if self.clip and dur>10 and align:
        #     # print('big file',dur)
        #     max_audio_start = int(dur - 10)
        #     audio_start = random.randint(0, max_audio_start)

        #     audio_norm = audio_norm[:,audio_start*sampling_rate:(audio_start+10)*sampling_rate]
        #     semb_ids = semb_ids[audio_start*50:(audio_start+10)*50 -1]

        #     86 mel -> 1s for 22050 setting
        # `   93 mel ->`1s for 24000 setting

        # add 64ms of values to start and end
        # audio_norm += torch.randn(audio_norm.shape[0])*1e-8
        # audio_norm = torch.concat([torch.randn(1412)*1e-8,audio_norm,torch.randn(1412)*1e-8])
        # audio_norm = audio_norm.unsqueeze(0)
        # y = torch.autograd.Variable(audio_norm, requires_grad=False)

        # assert(torch.min(y.data) >= -1)
        # assert(torch.max(y.data) <= 1)
        # magnitudes, phases = self.stft_fn.transform(y)
        # magnitudes = magnitudes.data
        # mel_output = torch.matmul(self.mel_basis, magnitudes)
        # mel_output = dynamic_range_compression(mel_output)
        # melspec = torch.squeeze(mel_output, 0)
        # energy = torch.norm(magnitudes, dim=1).squeeze(0)
        # melspec,energy = mel_spectrogram(audio_norm)
        melspec = get_mel_spectrogram(audio_norm, sampling_rate).squeeze(0)
        energy = []
        # if align:
        #     return melspec,list(energy),semb_ids
        return melspec, list(energy)

    def __len__(self):
        if self.scale:
            return self.data_len
        return len(self.data)

    # def get_process_heads(self,):
    #     '''
    #     divide data and heads based on the batch_size and weights
    # '''

    # new_heads ={}
    # new_weights =[]
    # process_batch_size = config.ts_batch_size*config.ts_gradient_accumulation_steps
    # sm=0
    # for i,j in zip(self.heads,self.weights):

    #     if sm + j > process_batch_size:
    #         if sm+j == process_batch_size:
    #             new_heads[i] = self.heads[i]
    #             new_weights.append(j)
    #         else:
    #             new_heads[i] = self.heads[i][:len(self.heads[i])*(process_batch_size-sm)//process_batch_size]
    #             new_weights.append(process_batch_size-sm)
    #     else:
    #         new_heads[i] = self.heads[i]
    #         new_weights.append(j)

    # self.get_worker_heads()

    # old heads and weights
    # new_heads = {}
    # for i in self.heads:
    #     segment_size = (len(self.heads[i]) + self.total_processes - 1) // self.total_processes
    #     start_idx = self.process_id * segment_size
    #     end_idx = start_idx + segment_size

    #     if end_idx > len(self.heads[i]):
    #         # Create a list that wraps around to the beginning
    #         segment = self.heads[i][start_idx:] + self.heads[i][:end_idx - len(self.heads[i])]
    #     else:
    #         segment = self.heads[i][start_idx:end_idx]
    #     new_heads[i]=segment
    # self.heads = new_heads
    # print(self.process_id,[len(self.heads[i]) for i in self.heads])
    # self.get_worker_heads()

    def get_worker_heads(
        self,
    ):
        self.worker_id = get_worker_info().id
        self.num_worker = get_worker_info().num_workers
        new_heads = {}
        for i in self.heads:
            segment_size = (len(self.heads[i]) + self.num_worker - 1) // self.num_worker
            start_idx = self.worker_id * segment_size
            end_idx = start_idx + segment_size

            if end_idx > len(self.heads[i]):
                # Create a list that wraps around to the beginning
                segment = (
                    self.heads[i][start_idx:]
                    + self.heads[i][: end_idx - len(self.heads[i])]
                )
            else:
                segment = self.heads[i][start_idx:end_idx]
            new_heads[i] = segment
        self.heads = new_heads
        # print("worker:",self.worker_id,self.process_id,[len(self.heads[i]) for i in self.heads],self.weights)

    def get_head(self):
        # self.get_process_heads()
        self.get_worker_heads()
        # print("weights:",self.weights,[h for h in self.heads])
        self.indices = [0] * len(self.heads)
        # self.process_heads = [{i:self.heads[i][self.process_id:]}for i in self.heads]
        while True:
            for (
                n,
                (head, weight),
            ) in enumerate(zip(self.heads, self.weights)):
                # if process_id == 0:
                #     print(weight,head)
                for i in range(weight):
                    if self.indices[n] < len(self.heads[head]):
                        # print(self.heads[head][self.indices[n]],worker_id,self.indices)
                        yield self.heads[head][self.indices[n]]
                        self.indices[n] += 1
                    else:
                        self.indices[n] = 0
                        random.shuffle(self.heads[head])
                        # shuffle the indices

    def __getitem__(self, index) -> Any:
        if self.iterator is None:
            self.iterator = self.get_head()
        if not self.scale:
            lang, path, semb, text = self.data[index]
            ref_mels = self.ref_mels[path][: self.ref_k]

        else:
            # line = read_specific_line(self.transcript_path,index+1)

            index = next(self.iterator)
            # print(self.worker_id,self.process_id,index)
            self.mmapped_file.seek(self.line_index[index])
            line = self.mmapped_file.readline().decode("utf-8")

            lang, path, text, semb_ids, ref_mels = line.split("|")
            # a=5/0
            # semb_ids = [int(i)+1 for i in semb_ids.split()]
            semb = semb_ids.split()
            ref_mels = [i.split(",") for i in ref_mels.split("\t")][: self.ref_k]

        if len(semb) < 25:
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
        # try:
        text_ids = [text_enc["<S>"]] + [text_enc[i] for i in text] + [text_enc["<E>"]]
        semb_ids = (
            [code_enc["<SST>"]] + [code_enc[i] for i in semb] + [code_enc["<EST>"]]
        )

        # except Exception as e:
        #     print(e)
        #     print(lang,path,text,index)
        #     exit
        # input_ids = text_ids+semb_ids
        # pad_length = config.t2s_position-(len(text_ids)+len(semb_ids))

        # token_type_ids = [0]*len(text_ids)+[1]*len(semb_ids)+[0]*pad_length
        # positional_ids = [i for i in range(len(text_ids))]+[i for i in range(len(semb_ids))]+[0]*pad_length
        # labels = [-100]*len(text_ids)+semb_ids+[-100]*pad_length
        # attention_mask = [1]*len(input_ids)+[0]*pad_length
        # input_ids += [tok_enc['<PAD>']]*pad_length

        def get_random_portion(mel, mask_lengths):
            clip = mask_lengths <= CLIP_LENGTH
            ref_mel = mel[:, :, :CLIP_LENGTH].clone()
            for n, z in enumerate(clip):
                if not z:
                    start = np.random.randint(0, mask_lengths[n].item() - CLIP_LENGTH)
                    ref_mel[n, :, :] = mel[n, :, start : start + CLIP_LENGTH].clone()
            return ref_mel

        try:
            ref_mels = [self.get_mel(path)[0] for path, score in ref_mels]
        except Exception as e:
            print(index, e)
            if index + 1 < self.data_len:
                return self.__getitem__(index + 1)
            return self.__getitem__(0)

        ref_c = []
        for i in range(self.ref_k):
            if ref_mels[i] is None:
                continue
            ref_c.append(ref_mels[i])

        if len(ref_c) == 0:
            # print('no refs worthy')
            if index + 1 < self.data_len:
                return self.__getitem__(index + 1)
            return self.__getitem__(0)

        if len(ref_c) != self.ref_k:
            # print('less refs found',len(ref_c))
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


# def get_padded_seq(sequences):

#     max_len=max([len(s) for s in sequences])
#     for i in range(len(sequences)):
#         sequences[i]=sequences[i]+tok_enc['<PAD>']*(max_len-len(sequences[i]))

#     return sequences


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
    # paths=[]
    ref_mels = []
    langs = []
    # ref_mels_length=[]

    for b in batch:
        text_ids.append(b["text_ids"])
        semb_ids.append(b["semb_ids"])
        # paths.append(b['path'])
        ref_mels.append(b["ref_mels"])
        langs.append(b["lang"])
        # ref_mels_length.append(b['ref_mel_length'])

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

    # print(mel_padded.shape,torch.tensor(code).shape,torch.tensor(mel_length),get_mask_from_lengths(torch.tensor(mel_length)))

    return (
        torch.tensor(text_ids),
        torch.tensor(code),
        torch.tensor(text_len),
        torch.tensor(code_len),
        ref_mels_padded,
        torch.tensor(langs),
    )


def get_dataset(transcript_path, get_process_id, total_processes):
    return semantic_dataset_batch(
        transcript_path,
        scale=True,
        process_id=get_process_id,
        total_processes=total_processes,
    )


if __name__ == "__main__":
    accelerator = Accelerator(
        gradient_accumulation_steps=config.ts_gradient_accumulation_steps
    )  # ,kwargs_handlers=[ddp_kwargs]) mixed_precision="fp16",

    get_process_id = accelerator.process_index
    total_processes = accelerator.num_processes

    # train_dataset_ = semantic_dataset_batch(config.data_path+'/transcript_train_20s_final_normalized_filtered.txt','../'+config.data_path+'/semt.txt','../'+config.data_path+'/ref_clips.pkl',
    #                                         scale=True,process_id=get_process_id,total_processes = total_processes)
    # train_dataset_ = semantic_dataset_batch(config.data_path+'/transcript_train_20s_final_normalized_filtered.txt','../'+config.data_path+'/semt.txt','../'+config.data_path+'/ref_clips.pkl',
    #                                         scale=True,process_id=get_process_id,total_processes = total_processes)
    # train_dataset_ = semantic_dataset_batch(config.data_path+'/transcript_train_20s_final_normalized_filtered.txt','../'+config.data_path+'/semt.txt','../'+config.data_path+'/ref_clips.pkl',
    #                                         scale=True,process_id=get_process_id,total_processes = total_processes)
    train_dataset_ = semantic_dataset_batch(
        config.data_path + "/transcript_train_20s_final_normalized_filtered.txt",
        "../" + config.data_path + "/semt.txt",
        "../" + config.data_path + "/ref_clips.pkl",
        scale=True,
        process_id=get_process_id,
        total_processes=total_processes,
    )
    # sampler = WeightedRandomSampler(
    #                                train_dataset_.weights,
    #                                train_dataset_.count,
    #                                replacement=False)
    train_dataset = DataLoader(
        train_dataset_,
        pin_memory=True,
        persistent_workers=True,
        num_workers=config.ts_num_workers,
        batch_size=config.ts_batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=collate,
        sampler=None,
    )
    print("batch", config.ts_batch_size)
    # val_dataset = DataLoader(semantic_dataset_batch(config.data_path+'/transcript_test_20_final_normalized.txt','../'+config.data_path+'/semt.txt','../'+config.data_path+'/ref_clips.pkl',scale=True,process_id=get_process_id,total_processes = total_processes),pin_memory=True,
    #                          persistent_workers=True,num_workers=2,batch_size=config.ts_batch_size,shuffle=True,drop_last=False,collate_fn=collate)

    train_dataloader = accelerator.prepare(train_dataset)
    # if accelerator.is_local_main_process:
    #     from IPython import embed
    #     embed()

    # checkiong the sampler working
    import math
    from collections import defaultdict

    def calculate_duration(code_len):
        return math.ceil(((code_len + 1) / 50) * 2) / 2

    sampling = defaultdict(int)
    dataset = []
    batch_data = {}
    batch = 0
    batch_data[batch] = defaultdict(int)
    for n, data in enumerate(tqdm(train_dataloader)):
        # break
        text_ids, code, text_len, code_len, ref_clips, langs = data
        #     print(text_ids)
        #     print('=====')
        #     # break
        for i, j in zip(code_len, text_ids):
            dur = calculate_duration(i - 2)
            # print(dur,i,code.shape)
            # sampling[calculate_duration(i)]+=1
            dataset.append(list(j.detach().cpu().numpy()))

            if dur > 19.5:
                batch_data[batch]["20_sentence"] += 1
                continue
            if dur <= 5:
                batch_data[batch]["5s"] += 1
                continue
            elif dur <= 10:
                batch_data[batch]["10s"] += 1
                continue
            elif dur <= 15:
                batch_data[batch]["15s"] += 1
                continue
            elif dur <= 20:
                batch_data[batch]["20s"] += 1
                continue
        # print(batch)
        if (n + 1) % config.ts_gradient_accumulation_steps == 0:
            batch += 1
            batch_data[batch] = defaultdict(int)
            # break
        # if n==20:
        #     break
    # # print(sampling)
    with open(
        f"Sampling_data_meta/sampling_{accelerator.process_index}.pkl", "wb"
    ) as file:
        pkl.dump(batch_data, file)
    with open(
        f"Sampling_data_meta/sampling_dataset_{accelerator.process_index}.pkl", "wb"
    ) as file:
        pkl.dump(dataset, file)
    print(batch_data[0])
    # # # return 0
