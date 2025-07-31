import os,sys,time,struct
import torch,torchaudio

# sys.path.append("S2A/bigvgan_v2_24khz_100band_256x")
# sys.path.append("S2A/")
# sys.path.append("T2S/")
# sys.path.append("hifi-gan/")

from S2A.inference import *
from S2A.diff_model import DiffModel
from T2S.autoregressive import TS_model
from T2S.mel_spec import get_mel_spectrogram
from Text import labels,text_labels,code_labels
from config import config

#code encdec
text_enc = {j:i for i,j in enumerate(text_labels)}
text_dec = {i:j for i,j in enumerate(text_labels)}

# text encdec
code_enc = {j:i for i,j in enumerate(code_labels)}
code_dec = {i:j for i,j in enumerate(code_labels)}

def create_wav_header(sample_rate = 24000, bits_per_sample=16, channels=1):
    # "RIFF" chunk descriptor
    chunk_id = b'RIFF'
    chunk_size = 0xFFFFFFFF  # Placeholder for chunk size (unknown during streaming)
    format = b'WAVE'
    
    # "fmt " sub-chunk (16 bytes for PCM format)
    subchunk1_id = b'fmt '
    subchunk1_size = 16  # PCM format
    audio_format = 1     # PCM = 1 (linear quantization)
    num_channels = channels
    sample_rate = sample_rate
    byte_rate = sample_rate * num_channels * bits_per_sample // 8
    block_align = num_channels * bits_per_sample // 8
    bits_per_sample = bits_per_sample
    
    # "data" sub-chunk
    subchunk2_id = b'data'
    subchunk2_size = 0xFFFFFFFF  # Placeholder for data size (unknown during streaming)

    # Pack the header into a byte object using struct
    header = struct.pack('<4sI4s4sIHHIIHH4sI',
                         chunk_id,
                         chunk_size,
                         format,
                         subchunk1_id,
                         subchunk1_size,
                         audio_format,
                         num_channels,
                         sample_rate,
                         byte_rate,
                         block_align,
                         bits_per_sample,
                         subchunk2_id,
                         subchunk2_size)
    
    return header


def get_processed_clips(ref_clips):
  frame_rate = 24000
  new_ref_clips = []
  for i in ref_clips:
    if '_proc.wav' in i:
      new_ref_clips.append(i)
      continue
    audio = AudioSegment.from_file(i)
    audio = audio.set_channels(1)
    audio = audio.set_frame_rate(frame_rate).set_sample_width(2)
    audio.export(i[:-4]+'_proc.wav',format='wav')
    new_ref_clips.append(i[:-4]+'_proc.wav')

  return new_ref_clips

def get_ref_mels(ref_clips):
    ref_mels = []
    for i in ref_clips:
        audio_norm,sampling_rate = torchaudio.load(i)
        ref_mels.append(get_mel_spectrogram(audio_norm,sampling_rate).squeeze(0)[:, :500])

    ref_mels_padded = (torch.randn((len(ref_mels), 100, 500))) * 1e-9
    for i, mel in enumerate(ref_mels):
        ref_mels_padded[i, :, : mel.size(1)] = mel
    return ref_mels_padded.unsqueeze(0)

def load_cfm(checkpoint,vocoder_checkpoint=None,device="cpu"):
    FM = BASECFM()
    if vocoder_checkpoint is None:
        hifi = bigvgan.BigVGAN.from_pretrained('nvidia/bigvgan_v2_24khz_100band_256x', use_cuda_kernel=False)
    else:
        hifi = bigvgan.BigVGAN.from_pretrained(vocoder_checkpoint, use_cuda_kernel=False)

    hifi.remove_weight_norm()
    hifi = hifi.eval().to(device)

    model = DiffModel(input_channels=100,
                 output_channels=100,
                 model_channels=512,
                 num_heads=8,
                 dropout=0.1,
                 num_layers=8,
                 enable_fp16=False,
                 condition_free_per=0.0,
                 multispeaker=True,
                 style_tokens=100,
                 training=False,
                 ar_active=False)
    
    model.load_state_dict(torch.load(checkpoint,map_location=torch.device('cpu'))['model'])
    model.eval().to(device)
    mu= torch.load(checkpoint,map_location=torch.device('cpu'))['norms']['mean_val']
    std = torch.load(checkpoint,map_location=torch.device('cpu'))['norms']['std']

    return FM,hifi,model,mu,std

def load_t2s_model(checkpoint,device):
    model = TS_model(n_embed= 1024, n_layer= 30, n_head = 16)
    model.load_state_dict(torch.load(checkpoint,map_location=torch.device('cpu'))['model'],strict=True)
    model.eval()
    model.to(device)
    model.init_gpt_for_inference()
    return model

def prepare_inputs(text,ref_clips_m1,ref_clips_m2,language,device):
    code_ids = [code_enc["<SST>"]]
    text_ids = (
                torch.tensor(
                    [text_enc["<S>"]]
                    + [text_enc[i] for i in text.strip()]
                    + [text_enc["<E>"]]
                )
                .to(device)
                .unsqueeze(0)
            )
    language = (
            torch.tensor(config.lang_index[language]).to(device).unsqueeze(0)
        )
    
    ref_mels_m1 = get_ref_mels(get_processed_clips(ref_clips_m1))
    ref_mels_m2 = get_ref_mels(get_processed_clips(ref_clips_m2))

    return text_ids,code_ids,language,ref_mels_m1,ref_mels_m2
    

def infer(m1,m2,vocoder,FM,mu,std,text_ids,code_ids,language,ref_mels_m1,ref_mels_m2,device):
    with torch.no_grad():
        cond_latents = m1.get_speaker_latent(ref_mels_m1.to(device))
        code_emb = m1.generate(
                language.to(device), cond_latents.to(device), text_ids.to(device), code_ids, **{
                "temperature": 0.8,
                "length_penalty": None,
                "repetition_penalty": None,
                "top_k": 50,
                "top_p": 0.8,
                "do_sample": True,
                "num_beams": 1,
                "max_new_tokens": 1500
            }
            )[:, :-1]

        mel = FM(m2, code_emb+1, (1, 100, int(1+93*(code_emb.shape[-1]+1)/50)), ref_mels_m2.to(device), n_timesteps=20, temperature=1.0)
        mel = denormalize_tacotron_mel(mel,mu,std)
        audio = vocoder(mel)
        audio = audio.squeeze(0).detach().cpu()
        audio = audio * 32767.0
        audio = (
            audio.numpy().reshape(-1).astype(np.int16)
        )

    return audio

if __name__ == '__main__':
    os.makedirs("generated_samples/",exist_ok=True)
    start = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    m1_checkpoint = "pretrained_checkpoint/m1_gemma_benchmark_1_latest_weights.pt"
    m2_checkpoint = "pretrained_checkpoint/m2.pt"
    vocoder_checkpoint = 'pretrained_checkpoint/700_580k_multilingual_infer_ready/'

    FM,vocoder,m2,mu,std = load_cfm(m2_checkpoint,vocoder_checkpoint,device)
    m1 = load_t2s_model(m1_checkpoint,device)
    
    model_loading_time = time.time()

    output_file_path = "test.wav"
    
    texts = ["यह एक उदाहरणात्मक हिंदी पाठ है जिसका उद्देश्य भाषा की संरचना और शब्दों के प्रवाह को समझना है। भारत एक विविधताओं से भरा देश है जहाँ अनेक भाषाएँ, धर्म, और संस्कृतियाँ एक साथ मिलकर रहते हैं। यहाँ की परंपराएँ, त्योहार और भोजन इसकी सांस्कृतिक समृद्धि को दर्शाते हैं।"]
    languages=['hindi']

    ref_clips = [
        'speakers/female1/train_hindifemale_02794.wav',
        'speakers/female1/train_hindifemale_04167.wav',
        'speakers/female1/train_hindifemale_02795.wav'
        ]

    for n,(lang,text) in tqdm(enumerate(zip(languages,texts))):
        text_ids,code_ids,language,ref_mels_m1,ref_mels_m2 = prepare_inputs(text.lower(),
                                                                        ref_clips_m1=ref_clips,
                                                                        ref_clips_m2=ref_clips,
                                                                        language=lang
                                                                        ,device=device)
        audio_wav = infer(m1,m2,vocoder,FM,mu,std,text_ids,code_ids,language,ref_mels_m1,ref_mels_m2,device)

        with open(f"generated_samples/{n}_{lang}.wav",'wb') as file:
            file.write(create_wav_header(sample_rate = 24000, bits_per_sample=16, channels=1))
            file.write(audio_wav.tobytes())
    
    audio_generation_time=time.time()

    print()
    print(text)
    print(audio_generation_time-start,":Total Time taken")
    print(model_loading_time-start, ":Model Loading time")
    print(audio_generation_time-model_loading_time, ":Audio Generation time")