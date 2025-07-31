import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autocast

from config import config

from .modules import GST, AttentionBlock, mySequential, normalization


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=half, dtype=torch.float32)
        / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class TimestepBlock(nn.Module):
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    def forward(self, x, emb):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x


class QuartzNetBlock(TimestepBlock):
    """Similar to Resnet block with Batchnorm and dropout, and using Separable conv in the middle.
    if its the last layer,set se = False and separable = False, and use a projection layer on top of this.
    """

    def __init__(
        self,
        nin,
        nout,
        emb_channels,
        kernel_size=3,
        dropout=0.1,
        R=1,
        se=True,
        ratio=8,
        separable=False,
        bias=True,
        use_scale_shift_norm=True,
    ):
        super(QuartzNetBlock, self).__init__()
        self.use_scale_shift_norm = use_scale_shift_norm
        self.se = se
        self.in_layers = mySequential(
            nn.Conv1d(nin, nout, kernel_size=1, padding="same", bias=bias),
            nn.SiLU(),
            normalization(nout),
        )

        if nin == nout:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Conv1d(
                nin, nout, kernel_size=1, padding="same", bias=bias
            )

        nin = nout
        self.model = nn.Sequential(
            nn.Conv1d(nin, nout, kernel_size, padding="same"),
            nn.SiLU(),
            normalization(nout),
            nn.Dropout(p=dropout),
        )

        self.emb_layers = nn.Sequential(
            nn.Linear(
                emb_channels,
                2 * nout if use_scale_shift_norm else nout,
            ),
            nn.SiLU(),
        )

    def forward(self, x, emb, mask=None):
        x_new = self.in_layers(x)
        emb = self.emb_layers(emb)
        while len(emb.shape) < len(x_new.shape):
            emb = emb[..., None]
        scale, shift = torch.chunk(emb, 2, dim=1)
        x_new = x_new * (1 + scale) + shift
        y = self.model(x_new)

        return y + self.residual(x)


class QuartzAttn(TimestepBlock):
    def __init__(self, model_channels, dropout, num_heads):
        super().__init__()
        self.resblk = QuartzNetBlock(
            model_channels,
            model_channels,
            model_channels,
            dropout=dropout,
            use_scale_shift_norm=True,
        )
        self.attn = AttentionBlock(
            model_channels, num_heads, relative_pos_embeddings=True
        )

    def forward(self, x, time_emb):
        y = self.resblk(x, time_emb)
        return self.attn(y)


class QuartzNet9x5(nn.Module):
    def __init__(self, model_channels, num_heads, dropout=0.1, enable_fp16=False):
        super(QuartzNet9x5, self).__init__()
        self.enable_fp16 = enable_fp16
        kernels = [3] * 10
        quartznet = []
        attn = []
        for i in kernels:
            quartznet.append(
                QuartzNetBlock(
                    model_channels,
                    model_channels,
                    model_channels,
                    kernel_size=i,
                    dropout=dropout,
                    R=5,
                    se=True,
                )
            )
            attn.append(
                AttentionBlock(model_channels, num_heads, relative_pos_embeddings=True)
            )

        self.quartznet = nn.ModuleList(quartznet)
        self.attn = nn.ModuleList(attn)
        self.conv2 = nn.ModuleList(
            [
                QuartzNetBlock(
                    model_channels,
                    model_channels,
                    model_channels,
                    kernel_size=3,
                    dropout=dropout,
                    R=3,
                    separable=False,
                )
                for i in range(3)
            ]
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(model_channels, model_channels, 3, padding="same"),
            nn.SiLU(),
            normalization(model_channels),
            nn.Conv1d(model_channels, 100, 1, padding="same"),
        )

    def forward(self, x, time_emb):
        for n, (layer, attn) in enumerate(zip(self.quartznet, self.attn)):
            x = layer(x, time_emb)  # 256 dim
            x = attn(x)
        for layer in self.conv2:
            x = layer(x, time_emb)

        x = self.conv3(x)
        return x


class DiffModel(nn.Module):
    def __init__(
        self,
        input_channels=80,
        output_channels=160,
        model_channels=256,
        num_heads=8,
        dropout=0.1,
        num_layers=8,
        multispeaker=True,
        style_tokens=100,
        enable_fp16=False,
        condition_free_per=0.1,
        training=False,
        ar_active=False,
        in_latent_channels=10004,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.model_channels = model_channels
        self.output_channels = output_channels
        self.num_heads = num_heads
        self.dropout = dropout
        self.num_layers = num_layers
        self.enable_fp16 = enable_fp16
        self.condition_free_per = condition_free_per
        self.training = training
        self.multispeaker = multispeaker
        self.ar_active = ar_active
        self.in_latent_channels = in_latent_channels

        if not self.ar_active:
            self.code_emb = nn.Embedding(
                config.semantic_model_centroids + 1, model_channels
            )
            self.code_converter = mySequential(
                AttentionBlock(model_channels, num_heads, relative_pos_embeddings=True),
                AttentionBlock(model_channels, num_heads, relative_pos_embeddings=True),
                AttentionBlock(model_channels, num_heads, relative_pos_embeddings=True),
            )
        else:
            self.code_converter = mySequential(
                nn.Conv1d(
                    self.in_latent_channels, model_channels, 3, padding=1, bias=True
                ),
                AttentionBlock(model_channels, num_heads, relative_pos_embeddings=True),
                AttentionBlock(model_channels, num_heads, relative_pos_embeddings=True),
                AttentionBlock(model_channels, num_heads, relative_pos_embeddings=True),
                AttentionBlock(model_channels, num_heads, relative_pos_embeddings=True),
            )
        if self.multispeaker:
            self.GST = GST(
                model_channels, style_tokens, num_heads, in_channels=input_channels
            )

        self.code_norm = normalization(model_channels)
        self.time_norm = normalization(model_channels)
        self.code_time_norm = normalization(model_channels)

        self.time_embed = mySequential(
            nn.Linear(model_channels, model_channels),
            nn.SiLU(),
            nn.Linear(model_channels, model_channels),
        )

        self.input_block = nn.Conv1d(input_channels, model_channels, 3, 1, 1, bias=True)
        self.unconditioned_embedding = nn.Parameter(torch.randn(1, model_channels, 1))
        self.integrating_conv = nn.Conv1d(
            model_channels * 2, model_channels, kernel_size=1
        )

        self.code_time = TimestepEmbedSequential(
            QuartzAttn(model_channels, dropout, num_heads),
            QuartzAttn(model_channels, dropout, num_heads),
            QuartzAttn(model_channels, dropout, num_heads),
        )

        self.layers = QuartzNet9x5(
            model_channels, num_heads, self.enable_fp16, self.dropout
        )

    def get_speaker_latent(self, ref_mels):
        ref_mels = ref_mels.unsqueeze(1) if len(ref_mels.shape) == 3 else ref_mels

        conds = []
        for j in range(ref_mels.shape[1]):
            conds.append(self.GST(ref_mels[:, j, :, :]))

        conds = torch.cat(conds, dim=-1)
        conds = conds.mean(dim=-1)

        return conds.unsqueeze(2)

    def forward(
        self,
        x,
        t,
        code_emb,
        ref_clips=None,
        speaker_latents=None,
        conditioning_free=False,
    ):
        time_embed = self.time_norm(
            self.time_embed(
                timestep_embedding(t.unsqueeze(-1), self.model_channels)
            ).permute(0, 2, 1)
        ).squeeze(2)
        if conditioning_free:
            code_embed = self.unconditioned_embedding.repeat(x.shape[0], 1, x.shape[-1])
        else:
            if not self.ar_active:
                code_embed = self.code_norm(
                    self.code_converter(self.code_emb(code_emb).permute(0, 2, 1))
                )
            else:
                code_embed = self.code_norm(self.code_converter(code_emb))
        if self.multispeaker:
            assert speaker_latents is not None or ref_clips is not None
            if ref_clips is not None:
                speaker_latents = self.get_speaker_latent(ref_clips)
            cond_scale, cond_shift = torch.chunk(speaker_latents, 2, dim=1)
            code_embed = code_embed * (1 + cond_scale) + cond_shift

        if self.training and self.condition_free_per > 0:
            unconditioned_batches = (
                torch.rand((code_embed.shape[0], 1, 1), device=code_embed.device)
                < self.condition_free_per
            )
            code_embed = torch.where(
                unconditioned_batches,
                self.unconditioned_embedding.repeat(code_embed.shape[0], 1, 1),
                code_embed,
            )

        expanded_code_emb = F.interpolate(code_embed, size=x.shape[-1], mode="linear")

        x_cond = self.code_time_norm(self.code_time(expanded_code_emb, time_embed))

        x = self.input_block(x)
        x = torch.cat([x, x_cond], dim=1)
        x = self.integrating_conv(x)
        out = self.layers(x, time_embed)

        return out
