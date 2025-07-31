import functools
import os
import sys
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (GemmaConfig, GemmaModel, GPT2Config, GPT2LMHeadModel,
                          GPT2Model, GPT2Tokenizer)

from config import config
from Text import code_labels, labels, text_labels

from .gpt_inference import GPT2InferenceModel
from .t2s_modules import GST

# code encdec
text_enc = {j: i for i, j in enumerate(text_labels)}
text_dec = {i: j for i, j in enumerate(text_labels)}

# text encdec
code_enc = {j: i for i, j in enumerate(code_labels)}
code_dec = {i: j for i, j in enumerate(code_labels)}


def null_position_embeddings(range, dim):
    return torch.zeros((range.shape[0], range.shape[1], dim), device=range.device)


class TS_model(nn.Module):
    def __init__(
        self, n_embed=1024, n_layer=30, n_head=16, n_positions=config.t2s_position
    ):
        super(TS_model, self).__init__()
        assert (n_embed / n_head) % 2 == 0, "n_embed n_head not a division of 2"
        self.vocab_size = len(labels)
        self.n_positions = n_positions
        self.n_embed = n_embed
        self.n_layer = n_layer
        self.n_head = n_head

        if self.vocab_size % 2 != 0:
            self.vocab_size += 1
        k = 1

        self.config = GemmaConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.n_embed,
            intermediate_size=self.n_embed * k,
            num_hidden_layers=self.n_layer,
            num_attention_heads=self.n_head,
            num_key_value_heads=self.n_head,
            head_dim=int(self.n_embed / self.n_head),
            hidden_act="gelu_pytorch_tanh",
            hidden_activation=None,
            max_position_embeddings=self.n_positions,
            initializer_range=0.02,
            rms_norm_eps=1e-06,
            use_cache=True,
            pad_token_id=0,
            eos_token_id=1,
            bos_token_id=2,
            tie_word_embeddings=True,
            rope_theta=10000.0,
            attention_bias=False,
            attention_dropout=0.0,
        )

        self.gpt = GemmaModel(self.config)
        del self.gpt.embed_tokens

        self.GST = GST(
            model_channels=self.n_embed,
            num_heads=self.n_head,
            in_channels=config.n_mel_channels,
        )
        self.text_head = nn.Linear(self.n_embed, len(text_labels))
        self.code_head = nn.Linear(self.n_embed, len(code_labels))

        self.text_embed = nn.Embedding(len(text_labels), self.n_embed)
        self.code_embed = nn.Embedding(len(code_labels), self.n_embed)
        self.language_embed = nn.Embedding(len(config.lang_index), self.n_embed)
        self.final_norm = nn.LayerNorm(self.n_embed)

    def init_random_embeddings(self):
        self.text_embed.weight.data.uniform_(-1, 1)
        self.code_embed.weight.data.uniform_(-1, 1)

    def get_speaker_latent(self, ref_mels):
        ref_mels = ref_mels.unsqueeze(1) if len(ref_mels.shape) == 3 else ref_mels

        conds = []
        for j in range(ref_mels.shape[1]):
            conds.append(self.GST(ref_mels[:, j, :, :]))

        conds = torch.cat(conds, dim=-1)
        conds = conds.mean(dim=-1)

        return conds.unsqueeze(1)

    def forward(
        self,
        text_ids,
        codes_ids=None,
        speaker_embed=None,
        ref_clips=None,
        language=torch.tensor(0),
        attn_mask=None,
        return_loss=False,
    ):
        assert speaker_embed is not None or ref_clips is not None
        text_embed = self.text_embed(text_ids)

        lanugage_embed = self.language_embed(language).unsqueeze(1)
        code_embed = None
        code_probs = None

        if codes_ids is not None:
            code_embed = self.code_embed(codes_ids)

        if ref_clips is not None:
            speaker_embed = self.get_speaker_latent(ref_clips)

        text_embed, code_embed = self.get_logits(
            lanugage_embed=lanugage_embed,
            speaker_embed=speaker_embed,
            text_embed=text_embed,
            code_embed=code_embed,
            attn_mask=attn_mask,
        )
        text_probs = self.text_head(text_embed).permute(0, 2, 1)

        if codes_ids is not None:
            code_probs = self.code_head(code_embed).permute(0, 2, 1)

        if return_loss:
            loss_text = F.cross_entropy(
                text_probs[:, :, :-1], text_ids[:, 1:].long(), reduce=False
            )
            loss_mel = F.cross_entropy(
                code_probs[:, :, :-1], codes_ids[:, 1:].long(), reduce=False
            )
            return loss_text, loss_mel, code_probs

        return text_probs, code_probs

    def get_logits(
        self, lanugage_embed, speaker_embed, text_embed, code_embed=None, attn_mask=None
    ):
        if code_embed is not None:
            embed = torch.cat(
                [lanugage_embed, speaker_embed, text_embed, code_embed], dim=1
            )
            position_ids = torch.zeros(
                (embed.shape[0], embed.shape[1]), device=embed.device
            )
            indices = torch.tensor(
                [0, 0]
                + list(range(text_embed.shape[1]))
                + list(range(code_embed.shape[1])),
                device=embed.device,
            )
            position_ids[:, : indices.size(0)] = indices
        else:
            embed = torch.cat([lanugage_embed, speaker_embed, text_embed], dim=1)
            position_ids = torch.zeros(
                (embed.shape[0], embed.shape[1]), device=embed.device
            )
            indices = torch.tensor(
                [0, 0] + list(range(text_embed.shape[1])), device=embed.device
            )
            position_ids[:, : indices.size(0)] = indices

        if attn_mask is None:
            attn_mask = torch.ones_like(embed).to(embed.device)
        else:
            attn_mask = torch.cat(
                [torch.ones((embed.shape[0], 2)).to(embed.device), attn_mask], dim=1
            )
        gpt_output = self.gpt(
            inputs_embeds=embed,
            attention_mask=attn_mask,
            position_ids=position_ids,
            return_dict=True,
        )
        enc = gpt_output.last_hidden_state[:, 2:]
        enc = self.final_norm(enc)
        if code_embed is not None:
            return enc[:, : text_embed.shape[1]], enc[:, -code_embed.shape[1] :]

        return enc[:, : text_embed.shape[1]], None

    def init_gpt_for_inference(self, kv_cache=True, use_deepspeed=False):
        self.gpt_inference = GPT2InferenceModel(
            self.config,
            self.gpt,
            None,
            self.code_embed,
            self.final_norm,
            self.code_head,
            kv_cache=kv_cache,
        )
        self.gpt.embed_tokens = self.code_embed

        if use_deepspeed:
            import deepspeed

            self.ds_engine = deepspeed.init_inference(
                model=self.gpt_inference.half(),  # Transformers models
                mp_size=1,  # Number of GPU
                dtype=torch.float32,  # desired data type of output
                replace_method="auto",  # Lets DS autmatically identify the layer to replace
                replace_with_kernel_inject=True,  # replace the model with the kernel injector
            )
            self.gpt_inference = self.ds_engine.module.eval()

    def compute_embeddings(self, language, cond_latents, text_inputs, code_inputs):
        text_embed = self.text_embed(text_inputs)
        lanugage_embed = self.language_embed(language).unsqueeze(1)

        emb = torch.cat([lanugage_embed, cond_latents, text_embed], dim=1)

        position_ids = torch.zeros(
            (emb.shape[0], emb.shape[1] + len(code_inputs)), device=emb.device
        )
        indices = torch.tensor(
            [0, 0] + list(range(text_embed.shape[1])) + list(range(len(code_inputs))),
            device=emb.device,
        )
        position_ids[:, : indices.size(0)] = indices

        self.gpt_inference.store_prefix_emb(emb)
        gpt_inputs = torch.full(
            (
                emb.shape[0],
                emb.shape[1] + len(code_inputs),  # +1 for the start_audio_token
            ),
            fill_value=1,
            dtype=torch.long,
            device=text_inputs.device,
        )
        gpt_inputs[:, -len(code_inputs) :] = torch.tensor(code_inputs)
        return (gpt_inputs, position_ids)

    def generate(
        self,
        language,
        cond_latents,
        text_inputs,
        code_inputs=[code_enc["<SST>"]],
        **hf_generate_kwargs,
    ):
        gpt_inputs, position_ids = self.compute_embeddings(
            language, cond_latents, text_inputs, code_inputs
        )
        gen = self.gpt_inference.generate(
            gpt_inputs,
            bos_token_id=code_enc["<SST>"],
            pad_token_id=code_enc["<PAD>"],
            eos_token_id=code_enc["<EST>"],
            max_length=self.n_positions,
            position_ids=position_ids,
            **hf_generate_kwargs,
        )
        if "return_dict_in_generate" in hf_generate_kwargs:
            return gen.sequences[:, gpt_inputs.shape[1] :], gen
        return gen[:, gpt_inputs.shape[1] - len(code_inputs) + 1 :]

    def get_generator(self, fake_inputs, **hf_generate_kwargs):
        return self.gpt_inference.generate_stream(
            fake_inputs,
            bos_token_id=code_enc["<SST>"],
            pad_token_id=code_enc["<PAD>"],
            eos_token_id=code_enc["<EST>"],
            max_length=self.n_positions,
            do_stream=True,
            **hf_generate_kwargs,
        )


class LearnedPositionEmbeddings(nn.Module):
    def __init__(self, seq_len, model_dim, init=0.02):
        super().__init__()
        self.emb = nn.Embedding(seq_len, model_dim)
        self.emb.weight.data.normal_(mean=0.0, std=init)

    def forward(self, x):
        sl = x.shape[1]
        return self.emb(torch.arange(0, sl, device=x.device))

    def get_fixed_embedding(self, ind, dev):
        return self.emb(torch.tensor([ind], device=dev)).unsqueeze(0)
