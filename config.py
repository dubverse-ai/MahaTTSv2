import os,torch
import sys

dir_path = os.path.abspath(os.path.dirname(__file__))

class config:

    data_path='./'
    model_name="OSS Shogun"
    user_name = "xxx" #for wandb
    
    desc = '''
        OSS
    '''

    train_file = "Semantic_tokens/SEMANTICS_/train.txt" 
    val_file = "Semantic_tokens/SEMANTICS_/val.txt"

    norms = torch.load(f"{dir_path}/mel_norms.pt")
    mu = norms["mean_val"]
    std = norms["std"]
    scale = True
    semantic_model_centroids = 10000 + 1
    seed_value = 3407

    t2s_checkpoint = "/omega/Models/FT_ENGLISH/T2S/1_latest.pt"
    ts_finetuning = True
    ts_wandb_logs = False
    text_loss_weight = 0.01
    t2s_position = 8192
    ts_batch_size = 1
    ts_epochs = 10
    ts_lr = 1e-5
    ts_weight_decay = 1e-4
    ts_eval_epoch = 1
    ts_num_workers = 8
    ts_gradient_accumulation_steps = 1  # EfBS of 128 for finetuning, 256 for pretraining, around 9k steps for sft for 2 epochs
    ts_eval_step = 10000

    langs = [
        "odia",
        "assamese",
        "thai",
        "gujrati",
        "russian",
        "japanese",
        "punjabi",
        "hindi",
        "manipuri",
        "korean",
        "bhojpuri",
        "sanskrit",
        "english",
        "french",
        "bodo",
        "malayalam",
        "telugu",
        "kannada",
        "dogri",
        "marathi",
        "german",
        "italian",
        "rajasthani",
        "spanish",
        "arabic",
        "urdu",
        "gujarati",
        "tamil",
        "bengali",
    ]

    lang_index = {i: j for j, i in enumerate(langs)}

    # Train s2a
    sa_wandb_logs = False
    joint_training = (False,)  # doesn't work
    checkpoint = "/omega/Models/" + "FT_ENGLISH/"
    sa_timesteps_max = 1000
    sa_batch_size = 32
    sa_epochs = 5000000
    gradient_accumulation_steps = 4
    sa_lr = 1e-4
    sa_weight_decay = 1e-2
    sa_eval_step = 10000
    sa_infer = True
    sa_infer_epoch = 1
    sa_num_workers = 24

    # Train Dvae (not using)
    dvae_wandb_logs = True
    dvae_batch_size = 128
    dvae_epochs = 5000
    dvae_lr = 3e-4
    dvae_weight_decay = 1e-2
    dvae_eval_epoch = 1
    dvae_infer = True
    dvae_infer_epoch = 1
    dvae_num_workers = 16

    # Acoustic Properties, Do not change
    CLIP_LENGTH = 500
    MAX_WAV_VALUE = 32768.0 - 1
    filter_length = 1024
    hop_length = 256  # 256
    window = "hann"
    win_length = 1024
    n_mel_channels = 100
    sampling_rate = 24000
    mel_fmin = 0.0
    mel_fmax = None
    normalize = True
