import argparse
import glob
import math
import multiprocessing as mp
import os
import statistics
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd
import soundfile as sf
import torchaudio
from pydub import AudioSegment
from tqdm import tqdm


def check_sample_rate(file_path):
    try:
        info = sf.info(file_path)
        if info.samplerate != 24000:
            return file_path
    except Exception as e:
        return None  # In case of error, return None


def process_files(file_list):
    with Pool() as pool:
        result = list(
            tqdm(pool.imap(check_sample_rate, file_list), total=len(file_list))
        )
    return [file for file in result if file is not None]


def read_paths_from_file(filename):
    with open(filename, "r", encoding="utf8") as file:
        path_list = []
        for i in tqdm(file):
            path = i.split("|")[1]
            path_list.append(path.strip("\n"))

    return path_list[:]


def gather_paths_from_glob():
    return glob.glob("./**/*.wav", recursive=True)


def detect_leading_silence(sound, silence_threshold=-50, chunk_size=64):
    trim_ms = 0
    assert chunk_size > 0
    while sound[
        trim_ms : trim_ms + chunk_size
    ].dBFS < silence_threshold and trim_ms < len(sound):
        trim_ms += chunk_size
    return trim_ms


def preprocess_audio(path, target_dBFS, frame_rate):
    durations = []
    dbfs = []
    audio = AudioSegment.from_file(path)
    dbfs.append(audio.dBFS)
    audio = audio.set_channels(1)
    audio = audio.set_frame_rate(frame_rate).set_sample_width(2)

    start_trim = detect_leading_silence(audio)
    end_trim = detect_leading_silence(audio.reverse())

    duration = len(audio)
    audio = audio[start_trim : duration - end_trim]
    audio = (
        AudioSegment.silent(duration=256, frame_rate=22050)
        + audio
        + AudioSegment.silent(duration=256, frame_rate=22050)
    )

    if path[-4:] == ".wav":
        audio.export(path[:-4] + ".wav", format="wav")
    elif path[-5:] == ".flac":
        audio.export(path[:-5] + ".flac", format="flac")
    else:
        audio.export(path[:-4] + ".wav", format="wav")

    durations.append(audio.duration_seconds)

    return dbfs, durations


def preprocess_audio_chunk(args):
    path_list_chunk, target_dBFS, frame_rate, n = args
    dbfs = []
    durations = []
    for i in tqdm(path_list_chunk, desc="preprocess " + str(n)):
        try:
            audio = AudioSegment.from_file(i)
            dbfs_i, durations_i = preprocess_audio(i, target_dBFS, frame_rate)
            dbfs.extend(dbfs_i)
            durations.extend(durations_i)
        except Exception as e:
            print(n, i, e)

    return dbfs, durations


def preprocess_audio_paths(path_list, target_dBFS, frame_rate, num_workers):
    chunk_size = len(path_list) // num_workers
    path_chunks = [
        path_list[i : i + chunk_size] for i in range(0, len(path_list), chunk_size)
    ]

    with Pool(num_workers) as pool:
        results = pool.map(
            preprocess_audio_chunk,
            [
                (chunk, target_dBFS, frame_rate, n)
                for n, chunk in enumerate(path_chunks)
            ],
        )

    dbfs = []
    durations = []
    for dbfs_i, durations_i in results:
        dbfs.extend(dbfs_i)
        durations.extend(durations_i)
    return dbfs, durations


def gather_metadata_chunk(args):
    path_list_chunk = args
    dbfs = []
    durations = []
    files = []
    for i in tqdm(path_list_chunk):
        try:
            path = i.split("|")[0]
            audio = AudioSegment.from_file(path)
            if audio.dBFS == -math.inf:
                print("=====================")
                print(path)
                print("=====================")
                continue
            dbfs.append(audio.dBFS)
            durations.append(audio.duration_seconds)
            files.append((audio.duration_seconds, i))
            if audio.duration_seconds == 0:
                print(i)
        except Exception as e:
            print(e, i)

    return dbfs, durations, files


def gather_metadata(path_list, num_workers):
    chunk_size = len(path_list) // num_workers
    path_chunks = [
        path_list[i : i + chunk_size] for i in range(0, len(path_list), chunk_size)
    ]

    with Pool(num_workers) as pool:
        results = pool.map(gather_metadata_chunk, [chunk for chunk in path_chunks])

    dbfs = []
    durations = []
    files = []
    for dbfs_i, durations_i, files_i in results:
        dbfs.extend(dbfs_i)
        durations.extend(durations_i)
        files.extend(files_i)

    files = sorted(files, key=lambda x: x[0])
    with open(os.path.join(data_dir, "files_duration.txt"), "w") as file:
        file.write("\n".join([i[1] + "|" + str(i[0]) for i in files]))

    with open(os.path.join(data_dir, "files.txt"), "w") as file:
        file.write("\n".join([i[1] for i in files if 2.0 < float(i[0]) < 15.0]))

    return dbfs, durations


def process_audio_data(input_file, mode, num_workers, data_dir):
    if input_file:
        path_list = read_paths_from_file(input_file)
    else:
        path_list = gather_paths_from_glob()

    speakers = []
    for n, i in enumerate(path_list):
        try:
            speakers.append(i.split("/")[-2])
        except:
            print(n, i)

    print("total audio files:", len(path_list))

    if mode == "preprocess":
        print("Preprocessing!")
        target_dBFS = -24.196741  # not using

        frame_rate = 24000
        dbfs, durations = preprocess_audio_paths(
            path_list[:], target_dBFS, frame_rate, num_workers
        )

        print("min duration : ", min(durations))
        print("max duration : ", max(durations))
        print("avg duration : ", sum(durations) / len(durations))
        print("Standard Deviation of durations % s" % (statistics.stdev(durations)))
        print("total duration : ", sum(durations))
        print("DONE")

    if mode == "metadata":
        print("Gathering metadata")
        dbfs, durations = gather_metadata(path_list[:], num_workers)

        print("min duration : ", min(durations))
        print("max duration : ", max(durations))
        print("avg duration : ", sum(durations) / len(durations))
        print("total duration : ", sum(durations))
        print("Standard Deviation of sample is % s" % (statistics.stdev(durations)))
        print("DONE")

        # pd.DataFrame({'dBFS': dbfs, 'duration': durations, 'files': [i[1] for i in files]}).to_csv("meta.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Audio processing script for preprocessing and metadata gathering",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Preprocess audio files from a text file
  python convert_factorize.py --input paths.txt --mode preprocess
  
  # Gather metadata from audio files in current directory
  python convert_factorize.py --mode metadata
  
  # Preprocess audio files found recursively in current directory
  python convert_factorize.py --mode preprocess
        """
    )
    
    parser.add_argument(
        "--input", "-i",
        type=str,
        help="Path to text file containing audio file paths in 'language|abspath|text' format"
    )
    
    parser.add_argument(
        "--mode", "-m",
        type=str,
        choices=["preprocess", "metadata"],
        required=True,
        help="Processing mode: 'preprocess' to process audio files, 'metadata' to gather statistics"
    )
    
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=4,
        help="Number of worker processes for parallel processing (default: 4)"
    )
    
    args = parser.parse_args()

    data_dir = "/".join(args.input.split("/")[:-1])
    print(f"Data directory: {data_dir}")
    
    print(f"Input file: {args.input}")
    print(f"Mode: {args.mode}")
    print(f"Number of workers: {args.workers}")
    
    process_audio_data(args.input, args.mode, args.workers, data_dir)
