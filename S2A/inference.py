import os
import sys

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../bigvgan_v2_24khz_100band_256x/")
    )
)

import bigvgan
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pydub import AudioSegment
from tqdm import tqdm

from config import config

from .flow_matching import BASECFM
from .utilities import denormalize_tacotron_mel, normalize_tacotron_mel


def infer(model, timeshapes, code_embs, ref_mels, epoch=0):
    os.makedirs("Samples/" + config.model_name + "/S2A/", exist_ok=True)
    FM = BASECFM()
    device = next(model.parameters()).device

    hifi = bigvgan.BigVGAN.from_pretrained(
        "nvidia/bigvgan_v2_24khz_100band_256x", use_cuda_kernel=False
    )
    hifi.remove_weight_norm()
    hifi = hifi.eval().to(device)

    audio_paths = []
    mels = []
    for n, (timeshape, code_emb, ref_mel) in enumerate(
        zip(timeshapes, code_embs, ref_mels)
    ):
        with torch.no_grad():
            mel = FM(
                model,
                code_emb.unsqueeze(0).to(device),
                (1, 100, timeshape),
                ref_mel.unsqueeze(0).to(device),
                n_timesteps=20,
                temperature=1.0,
            )
            mel = denormalize_tacotron_mel(mel)
            mels.append(mel)
            audio = hifi(mel)
            audio = audio.squeeze(0).detach().cpu()
            audio = audio * 32767.0
            audio = audio.numpy().reshape(-1).astype(np.int16)

        audio_path = (
            "../Samples/"
            + config.model_name
            + "/S2A/"
            + str(epoch)
            + "_"
            + str(n)
            + ".wav"
        )
        AudioSegment(
            audio.tobytes(),
            frame_rate=24000,
            sample_width=audio.dtype.itemsize,
            channels=1,
        ).export(audio_path, format="wav")
        audio_paths.append(audio_path)

    return audio_paths, mels
