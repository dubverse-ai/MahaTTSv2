<div align="center">

<a href="https://ibb.co/wN1LS7K"><img width="320" height="173" alt="Screenshot-2024-01-15-at-8-14-08-PM" src="https://github.com/user-attachments/assets/af22f00d-e9d6-49e1-98b1-7efeac900f9a" /></a>

<h1>MahaTTS v2: An Open-Source Large Speech Generation Model</h1>
a <a href = "https://black.dubverse.ai">Dubverse Black</a> initiative <br> <br>

<!-- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1qkZz2km-PX75P0f6mUb2y5e-uzub27NW?usp=sharing) -->
</div>

------
## Description
We introduce MahaTTS v2, a multi-speaker text-to-speech (TTS) system that has been trained on 50k hours of Indic and global languages. 
We have followed a text-to-semantic-to-acoustic approach, leveraging wav2vec2 tokens, this gives out-the-box generalization to unseen low-resourced languages. 
We have open sourced the first version (MahaTTS), which was trained on English and Indic languages as two separate models on 9k and 400 hours of open source datasets. 
In MahaTTS v2, we have collected over 20k+ hours of training data into a single multilingual cross-lingual model. 
We have used gemma as the backbone for text-to-semantic modeling and a conditional flow model for semantics to mel spectogram generation, using a BigVGAN vocoder to generate the final audio waveform. 
The model has shown great robustness and quality results compared to the previous version.
We are also open sourcing the ability to finetune on your own voice.

### With this release:
- generate voices in multiple seen and unseen speaker identities (voice cloning)
- generate voices in multiple langauges (multilingual and cross-lingual voice cloning)
- copy the style of speech from one speaker to another (cross-lingual voice cloning with prosody and intonation transfer)
- Train your own large scale pretraining or finetuning Models.

### MahaTTS Architecture

<img width="1023" height="859" alt="Screenshot 2025-07-10 at 4 04 08 PM" src="https://github.com/user-attachments/assets/4d44cc35-4b66-41a1-b4fd-415af35eda87" />




## Installation

```bash
git lfs install
git clone --recurse-submodules https://huggingface.co/Dubverse/MahaTTSv2
pip install -r MahaTTSv2/requirements.txt
```

```bash
import sys
sys.path.append("MahaTTSv2/")
import os
import torch
import subprocess
from inference import infer, prepare_inputs, load_t2s_model, load_cfm, create_wav_header

device = "cuda"# if torch.cuda.is_available() else "cpu"
print("Using device", device)

# Model checkpoints
m1_checkpoint = "MahaTTSv2/pretrained_checkpoint/m1_gemma_benchmark_1_latest_weights.pt"
m2_checkpoint = "MahaTTSv2/pretrained_checkpoint/m2.pt"
vocoder_checkpoint = 'MahaTTSv2/pretrained_checkpoint/700_580k_multilingual_infer_ready/'

global FM, vocoder, m2, mu, std, m1

# Load models
FM, vocoder, m2, mu, std = load_cfm(m2_checkpoint, vocoder_checkpoint, device)
m1 = load_t2s_model(m1_checkpoint, device)


def generate_audio(text, language):

    ref_clips = [
        'speakers/female1/train_hindifemale_02794.wav',
        'speakers/female1/train_hindifemale_04167.wav',
        'speakers/female1/train_hindifemale_02795.wav'
        ]

    text_ids, code_ids, language_code, ref_mels_m1, ref_mels_m2 = prepare_inputs(
        text.lower(),
        ref_clips_m1=ref_clips,
        ref_clips_m2=ref_clips,
        language=language,
        device=device
    )

    audio_wav = infer(m1, m2, vocoder, FM, mu, std, text_ids, code_ids, language_code, ref_mels_m1, ref_mels_m2, device)
    return 24000,audio_wav

```


### Model Params
|      Model                | Parameters | Model Type |       Output      |  
|:-------------------------:|:----------:|------------|:-----------------:|
|   Text to Semantic (M1)   |    510 M   | Causal LM  |   10,001 Tokens   |
|  Semantic to MelSpec(M2)  |    71 M    |   FLOW     |   100x Melspec    |
|      BigVGAN Vocoder      |    112 M   |    GAN     |   Audio Waveform  |


## 🌐 Supported Languages

The following languages are currently supported:

| Language         | Status |
|------------------|:------:|
| Assamese (in)    | ✅     |
| Bengali (in)     | ✅     |
| Bhojpuri (in)    | ✅     |
| Bodo (in)        | ✅     |
| Dogri (in)       | ✅     |
| Odia (in)        | ✅     |
| English (en)     | ✅     |
| French (fr)      | ✅     |
| Gujarati (in)    | ✅     |
| German (de)      | ✅     |
| Hindi (in)       | ✅     |
| Italian (it)     | ✅     |
| Kannada (in)     | ✅     |
| Malayalam (in)   | ✅     |
| Marathi (in)     | ✅     |
| Telugu (in)      | ✅     |
| Punjabi (in)     | ✅     |
| Rajasthani (in)  | ✅     |
| Sanskrit (in)    | ✅     |
| Spanish (es)     | ✅     |
| Tamil (in)       | ✅     |
| Telugu (in)      | ✅     |


## TODO:
1. Addind Training Instructions.
2. Add a colab for the same.


## License
MahaTTS is licensed under the Apache 2.0 License. 

## 🙏 Appreciation

- [Tortoise-tts](https://github.com/neonbjb/tortoise-tts) for inspiring the architecture
- [M4t Seamless](https://github.com/facebookresearch/seamless_communication) [AudioLM](https://arxiv.org/abs/2209.03143) and many other ground-breaking papers that enabled the development of MahaTTS
- [BIGVGAN](https://github.com/NVIDIA/BigVGAN) out of the box vocoder
- [Flow training](https://github.com/shivammehta25/Matcha-TTS) for training Flow model
- [Huggingface](https://huggingface.co/docs/transformers/index) for related training and inference code
