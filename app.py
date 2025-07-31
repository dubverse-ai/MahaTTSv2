import sys, os, torch
# sys.path.append("Testing/")
import gradio as gr
from inference import infer, prepare_inputs, load_t2s_model, load_cfm, create_wav_header
from tqdm import tqdm

# Setup
os.makedirs("generated_samples/", exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device", device)

# Model checkpoints
m1_checkpoint = "pretrained_checkpoint/m1_gemma_benchmark_1_latest_weights.pt"
m2_checkpoint = "pretrained_checkpoint/m2.pt"
vocoder_checkpoint = 'pretrained_checkpoint/700_580k_multilingual_infer_ready/'

global FM, vocoder, m2, mu, std, m1

# Load models
FM, vocoder, m2, mu, std = load_cfm(m2_checkpoint, vocoder_checkpoint, device)
m1 = load_t2s_model(m1_checkpoint, device)


# Speaker reference clips
speaker_refs = {
    "Speaker1": [
            "speakers/female1/train_hindifemale_02794.wav",
            "speakers/female1/train_hindifemale_04167.wav",
            "speakers/female1/train_hindifemale_02795.wav"
        ]
}

# Available languages (can be extended)
available_languages = ["hindi"]

# Inference function
def generate_audio(text, speaker_name, language):
    if speaker_name not in speaker_refs:
        return f"Reference clips not available for {speaker_name}", None

    ref_clips = speaker_refs[speaker_name]   

    text_ids, code_ids, language_code, ref_mels_m1, ref_mels_m2 = prepare_inputs(
        text.lower(),
        ref_clips_m1=ref_clips,
        ref_clips_m2=ref_clips,
        language=language,
        device=device
    )

    audio_wav = infer(m1, m2, vocoder, FM, mu, std, text_ids, code_ids, language_code, ref_mels_m1, ref_mels_m2, device)
    return 24000,audio_wav

# Gradio UI
interface = gr.Interface(
    fn=generate_audio,
    inputs=[
        gr.Textbox(label="Enter Text"),
        gr.Dropdown(choices=list(speaker_refs.keys()), label="Select Speaker"),
        gr.Dropdown(choices=available_languages, label="Select Language")
    ],
    outputs=gr.Audio(label="Generated Speech"),
    title="MAHATTSv2 Demo",
    description="Enter text, choose a speaker and language to generate speech."
)

interface.launch(share=True,server_port=9999)
