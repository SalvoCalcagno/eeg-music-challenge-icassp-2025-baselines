import os
import mne
import json
import argparse
import numpy as np
from tqdm import tqdm
import src.config as config

mne.set_log_level('ERROR')

def parse():
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--split_dir', type=str, default=str("data/splits"))
    parser.add_argument('--input_dir', type=str, default=str("pruned"))
    parser.add_argument('--output_dir', type=str, default=str("preprocessed"))

    args = parser.parse_args()
    
    # add data_directory
    args.data_dir = config.get_attribute("dataset_path")
    
    return args

class ResidualNan(Exception):
    pass

def interpolate(raw_data):
    
    # replace very large values with nans
    raw_data[abs(raw_data) > 1e2] = np.nan

    # get indices of nans
    nan_indices = np.where(np.isnan(raw_data))
    nan_indices = np.vstack(nan_indices).transpose()

    # hypotesis, Punctual nans
    for channel, timepoint in nan_indices:

        # get value before the point
        before = raw_data[channel, timepoint-1]
        # get value after the point
        after = raw_data[channel, timepoint-1]

        # interpolate
        raw_data[channel, timepoint] = (before + after) / 2

    nan_indices = np.where(np.isnan(raw_data))
    nan_indices = np.vstack(nan_indices).transpose()
    any_nan = nan_indices.shape[0]!=0
    if any_nan:
        raise ResidualNan("Data still contain Nans after interpolation")
        
    return raw_data

def open_and_interpolate(file):
    raw_file = mne.io.read_raw_fif(file, preload=True)
    raw_data = raw_file.get_data()
    try:
        raw_data = interpolate(raw_data)
    except ResidualNan as e:
        print(f"Residual NaNs in {file}")
        return None
    return raw_data

def get_stats(file_list):
    tmp = []
    for file in tqdm(file_list):
        raw_data = open_and_interpolate(file)
        tmp.append(raw_data)
    # concatenate all the data
    data = np.concatenate(tmp, axis=1)
    #print(data.shape)
    # compute the mean and std
    mean = np.mean(data, axis=1)
    std = np.std(data, axis=1)
    return mean, std

def z_score(raw_data, mean, std):
    return (raw_data - mean[:, np.newaxis]) / std[:, np.newaxis]

def main(args):
    
    input_dir = os.path.join(args.data_dir, args.input_dir)
    output_dir = os.path.join(args.data_dir, args.output_dir)
    
    print(f"Input directory: {input_dir}")
    input_dirs = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, f))]
    # create the same directory structure in the output directory
    for d in input_dirs:
        print(f"Creating directory {os.path.join(output_dir, os.path.basename(d))}")
        os.makedirs(os.path.join(output_dir, os.path.basename(d)), exist_ok=True)
        
    # Create a list of input files to process
    print("Listing files...")
    files = []
    for dir in input_dirs:
        files.extend([os.path.join(dir, f) for f in os.listdir(dir) if f.endswith(".fif")])
    print(f"Found {len(files)} files to process!")
    
    print("Loading splits...")
    # Load the split file
    split_file_1 = os.path.join(args.split_dir, "splits_subject_identification.json")
    split_file_2 = os.path.join(args.split_dir, "splits_emotion_recognition.json")
    
    splits_1 = json.load(open(split_file_1, 'r'))
    splits_2 = json.load(open(split_file_2, 'r'))
    
    splits = {
        "train": splits_2["train"],
        "val_trial": splits_2["val_trial"],
        "val_subject": splits_2["val_subject"],
        "test_trial": splits_1["test_trial"],
        "test_subject": splits_2["test_subject"]

    }
    
    # Create a list with only train files for statistics
    train_files = [os.path.join(input_dir, "train", f"{s['id']}_eeg.fif") for s in splits["train"]]
    print(f"Found {len(train_files)} train files!")
    
    # Get global train statistics (per channel)
    print("Computing global statistics...")
    mean, std = get_stats(train_files)
    print("Global statistics computed!")
    
    print("Computing subject-wise statistics...")
    # Create a list with only train files for each subject for statistics
    train_files_per_subject = {}
    for file in splits["train"]:
        subject = file["subject_id"]
        if subject not in train_files_per_subject:
            train_files_per_subject[subject] = []
        train_files_per_subject[subject].append(os.path.join(input_dir, "train", f"{file['id']}_eeg.fif"))
    
    # Get train statistics per subject
    stats_per_subject = {
        subject_id : get_stats(files) for subject_id, files in train_files_per_subject.items()
    }
    print("Subject-wise statistics computed!")
    
    print("Preprocessing data...")
    # Process each file
    for file in tqdm(files):
        input_file = file
        output_file = file.replace(".fif", ".npy").replace(input_dir, output_dir)
        
        raw_data = open_and_interpolate(input_file)
        if raw_data is None:
            continue
        z_data = z_score(raw_data, mean, std)

        np.save(output_file, z_data)
    
    print("Preprocessing done!")
        
if __name__ == "__main__":
    args = parse()
    main(args)
    