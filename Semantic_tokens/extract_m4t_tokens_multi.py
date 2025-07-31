import argparse
import multiprocessing
import os
import random
import shutil
import sys
import time
from functools import partial

import numpy as np
import torch
import torchaudio
from tqdm import tqdm

from seamless_communication.models.unit_extractor import (
    KmeansModel, UnitExtractor, Wav2Vec2LayerOutputModel)


def train_test_split_large_file(input_file, train_file, test_file, test_ratio=0.2, seed=42):
    """
    Memory-efficient train-test split for large files.
    Performs two passes: first to count lines, second to split.
    """
    random.seed(seed)

    # First pass: count lines
    with open(input_file, "r", encoding="utf-8") as f:
        total_lines = sum(1 for _ in f)

    n_test = int(total_lines * test_ratio)

    # Choose random line numbers for test set
    test_indices = set(random.sample(range(total_lines), n_test))

    # Second pass: write to train/test files
    with (
        open(input_file, "r", encoding="utf-8") as in_f,
        open(train_file, "w", encoding="utf-8") as train_f,
        open(test_file, "w", encoding="utf-8") as test_f,
    ):
        for idx, line in enumerate(in_f):
            if idx in test_indices:
                test_f.write(line)
            else:
                train_f.write(line)

    print(
        f"Split {total_lines} lines into {total_lines - n_test} train and {n_test} test lines."
    )


lock = multiprocessing.Lock()


def process_data(data, device, out_layer_number, process_no, kmeans_uri, model_name, data_dir, batch_size=10000):
    lock.acquire()
    unit_extractor = UnitExtractor(
        model_name, kmeans_uri, device=torch.device(f"cuda:{device}")
    )
    lock.release()
    results = []
    # if i == 0:
    data = tqdm(data, desc="process no : " + str(process_no))
    for i in data:
        try:
            audio, sr = torchaudio.load(i[1])
            audio = (
                torchaudio.functional.resample(audio, sr, 16000)
                .squeeze(0)
                .unsqueeze(-1)
            )
            with torch.no_grad():
                units = (
                    unit_extractor.predict(audio.to(device), out_layer_number - 1)
                    .detach()
                    .cpu()
                    .numpy()
                )
            text = " ".join([str(k) for k in units]).strip()
            results.append("|".join(i) + "|" + text + "\n")
        except Exception as e:
            print(i, e)

        if len(results) == batch_size:
            semt = "".join(results)
            with open(os.path.join(data_dir, "SEMANTICS_/" + str(process_no) + "_semt.txt"), "a") as file:
                file.write(semt)
            results = []

    if len(results) != 0:
        semt = "".join(results)
        with open(os.path.join(data_dir, "SEMANTICS_/" + str(process_no) + "_semt.txt"), "a") as file:
            file.write(semt)
        results = []


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Extract M4T semantic tokens from audio files using multi-GPU processing"
    )
    
    parser.add_argument(
        "input_file",
        help="Path to the input file containing audio file paths and metadata"
    )
    
    parser.add_argument(
        "--kmeans-uri",
        default="https://dl.fbaipublicfiles.com/seamlessM4T/models/unit_extraction/kmeans_10k.npy",
        help="URI for the kmeans model (default: M4T kmeans model)"
    )
    
    parser.add_argument(
        "--model-name",
        default="xlsr2_1b_v2",
        help="Model name for unit extraction (default: xlsr2_1b_v2)"
    )
    
    parser.add_argument(
        "--out-layer-number",
        type=int,
        default=35,
        help="Output layer number for feature extraction (default: 35)"
    )
    
    parser.add_argument(
        "--gpu-multiplier",
        type=int,
        default=1,
        help="Multiplier for number of GPUs to use (default: 1)"
    )
    
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.1,
        help="Ratio of data to use for validation set (default: 0.1)"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10000,
        help="Number of results to accumulate before writing to file (default: 10000)"
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    start = time.time()
    
    # Parse command line arguments
    args = parse_arguments()
    
    # Extract parameters from arguments
    kmeans_uri = args.kmeans_uri
    model_name = args.model_name
    out_layer_number = args.out_layer_number
    k = args.gpu_multiplier
    test_ratio = args.test_ratio
    seed = args.seed
    batch_size = args.batch_size
    
    num_gpus_to_use = torch.cuda.device_count() * k
    data_dir = "/".join(args.input_file.split("/")[:-1])
    
    if os.path.exists(os.path.join(data_dir, "SEMANTICS_")):
        shutil.rmtree(os.path.join(data_dir, "SEMANTICS_"))
    os.makedirs(os.path.join(data_dir, "SEMANTICS_"), exist_ok=True)
    
    with open(args.input_file, "r") as file:
        data = file.read().strip("\n").split("\n")[:]

    data = [i.split("|") for i in data][:]

    # Split data into chunks for each GPU
    chunk_size = len(data) // num_gpus_to_use
    data_chunks = [data[i : i + chunk_size] for i in range(0, len(data), chunk_size)]
    if len(data_chunks) != num_gpus_to_use:
        data_chunks[-2] += data_chunks[-1]
        data_chunks = data_chunks[:-1]

    processes = []
    for i in range(num_gpus_to_use):
        p = multiprocessing.Process(
            target=process_data, 
            args=(data_chunks[i], i // k, out_layer_number, i, kmeans_uri, model_name, data_dir, batch_size)
        )
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    for i in range(num_gpus_to_use):
        with open(os.path.join(data_dir, "SEMANTICS_/" + str(i) + "_semt.txt"), "r") as file:
            data = file.read()

        with open(
            os.path.join(data_dir, "SEMANTICS_/" + os.path.basename(args.input_file).split(".")[0] + "_semt.txt"), "a"
        ) as file:
            file.write(data)

    input_file = os.path.join(data_dir, "SEMANTICS_/" + os.path.basename(args.input_file).split(".")[0] + "_semt.txt")
    train_file = (
        os.path.join(data_dir, "SEMANTICS_/" + os.path.basename(args.input_file).split(".")[0] + "_semt_train.txt")
    )
    test_file = (
        os.path.join(data_dir, "SEMANTICS_/" + os.path.basename(args.input_file).split(".")[0] + "_semt_val.txt")
    )

    train_test_split_large_file(
        input_file, train_file, test_file, test_ratio=test_ratio, seed=seed
    )
    print("processing took: with %d instances" % (num_gpus_to_use), time.time() - start)
    print(
        "data ready at:",
        os.path.join(data_dir, "SEMANTICS_/" + os.path.basename(args.input_file).split(".")[0] + "_semt.txt"),
    )
