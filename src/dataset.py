import os
import mne
import json
import numpy as np
from tqdm import tqdm
import src.config as config
from torchvision import transforms as T
from torch.utils.data import Dataset, DataLoader
from src.eeg_transforms import RandomCrop, ToTensor, Standardize

mne.set_log_level("ERROR")

class EremusDataset(Dataset):
    def __init__(self, subdir, split_dir, split="train", task="subject_identification", ext="fif", transform=None, prefix=""):
        
        self.dataset_dir = config.get_attribute("dataset_path", prefix=prefix)
        self.subdir = os.path.join(subdir, split) if "test" in split else os.path.join(subdir, "train")
        self.split_dir = split_dir
        self.transform = transform
        self.split = split
        self.label_name = "subject_id" if task == "subject_identification" else "label"
        self.ext = ext
        
        splits = json.load(open(os.path.join(split_dir, f"splits_{task}.json")))
        self.samples = splits[split]
        
        files = []
        for sample in self.samples:
            #path = os.path.join(self.dataset_dir, self.subdir, sample['filename_preprocessed'])
            path = os.path.join(self.dataset_dir, self.subdir, f"{sample['id']}_eeg.{self.ext}")
            files.append(path)
        files = list(set(files))
        #self.files = {f: np.load(f)['arr_0'] for f in files}
        if self.ext == "npy":
            self.files = {f: np.load(f) for f in tqdm(files)}
        elif self.ext == "fif":
            self.files = {f: mne.io.read_raw_fif(f, preload=True).get_data() for f in tqdm(files)}
        else:
            raise ValueError(f"Extension {ext} not recognized")
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        data = self.files[os.path.join(self.dataset_dir, self.subdir, f"{sample['id']}_eeg.{self.ext}")]
        sample = {
            "id": sample['id'],
            "eeg": data,
            "label": sample[self.label_name] if "test" not in self.split else -1,
        }
        if self.transform:
            sample = self.transform(sample)
        return sample    
      
def get_loaders(args):
    
    if args.task == "subject_identification":
        splits = ["train", "val_trial"]
    elif args.task == "emotion_recognition":
        splits = ["train", "val_trial", "val_subject"]
    else:
        raise ValueError(f"Task {args.task} not recognized")
    
    # Define transforms
    train_transforms = T.Compose([
        RandomCrop(args.crop_size),
        ToTensor(label_interface="long"),
        Standardize()
    ])
    
    test_transforms = T.Compose([
        ToTensor(label_interface="long"),
        Standardize()
    ])

    # Select dataset
    subdir = args.data_type
    if args.data_type == "raw":
        ext = "fif"
    elif args.data_type == "pruned":
        ext = "fif"
    else:
        ext = "npy"

    datasets = {
        split: EremusDataset(
            subdir=subdir,
            split_dir=args.split_dir,
            split=split,
            ext = ext,
            task = args.task,
            transform=train_transforms if split == "train" else test_transforms
        )
        for split in splits
    }
    
    
    loaders = {
        split: DataLoader(
            dataset,
            batch_size=args.batch_size if split == "train" else 1,
            shuffle=True if split == "train" else False,
            num_workers=args.num_workers
        )
        for split, dataset in datasets.items()
    }

    return loaders, args

def get_test_loader(args):
    
    if args.task == "subject_identification":
        splits = ["test_trial"]
    elif args.task == "emotion_recognition":
        splits = ["test_trial", "test_subject"]
    else:
        raise ValueError(f"Task {args.task} not recognized")
    
    # Define transforms
    test_transforms = T.Compose([
        ToTensor(label_interface="long"),
        Standardize()
    ])

    # Select dataset
    subdir = args.data_type
    if args.data_type == "raw":
        ext = "fif"
    elif args.data_type == "pruned":
        ext = "fif"
    else:
        ext = "npy"

    datasets = {
        split: EremusDataset(
        subdir=subdir,
        split_dir=args.split_dir,
        split=split,
        ext = ext,
        task = args.task,
        transform=test_transforms
        ) for split in splits
    }
    
    datasets_no_transform = {
        split: EremusDataset(
        subdir=subdir,
        split_dir=args.split_dir,
        split=split,
        ext = ext,
        task = args.task,
        transform=None
        ) for split in splits
    }
    
    loaders = {
        split: DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=args.num_workers
        )
        for split, dataset in datasets.items()
    }

    return datasets_no_transform, loaders, args