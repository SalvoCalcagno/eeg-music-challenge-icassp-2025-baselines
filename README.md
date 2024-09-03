<div align="center">

# Baselines for EEG-Music Emotion Recognition Grand Challenge at ICASSP 2025
Salvatore Calcagno, Simone Carnemolla, Isaak Kavasidis, Simone Palazzo, Daniela Giordano, Concetto Spampinato

</div>

# Overview
Baseline implementation for the <a href='https://eeg-music-challenge.github.io/eeg-music-challenge/'>EEG-Music Emotion Recognition Grand Challenge</a> hosted at <a href='https://2025.ieeeicassp.org/sp-grand-challenges/#gc5'>ICASSP2025</a>.

# Methods

## Preprocessing
We used pruned data as starting point and we applied minimal preprocessing involving:
- deletion of invalid values (nans and outliers)
- standardization with global per-channel statistics

## Architectures
We chose 3 well-known architectures as baselines for both tasks:
- EEGNet [1]
- SyncNet [2]
- EEGChannelNet [3] 
- 
No significant changes were applied to the original architectures.

## Training
For the subject identification task, we created a small validation set (val_trial) by extracting 2 trials per subject from the training data. For the emotion recognition task, before extracting the held-out-trial validation set, we selected 2 subjects to serve as a separate held-out-subject validation set (val_subject).

Models were trained using the Adam optimizer for 500 epochs. During training, the model was provided with a random window of 1280 timepoints. For validation, we first segmented each sample into smaller windows of 1280 timepoints, excluding the final segment. The model was then fed all the windows, and a voting scheme was applied to determine the final prediction.

The final model and hyperparameters (learning rate, batch size, voting scheme) were selected based on the highest  balanced accuracy on val_trial for the subject identification task and the average balanced accuracy across val_trial and val_subject, for the emotion recognition task. A grid search was conducted to optimize these parameters.

## Inference
For inference, same as for validation, each sample was first segmented into smaller windows of 1280 timepoints, excluding the final segment. 
The same voting scheme applied in validation was used to generate the final prediction.

# Results
Our strategy yields the following results that serve as baseline

| Model           | Subject Identification | Emotion Recognition |
|-----------------|------------------------|---------------------|
| EEGNet          | 65.91              | -                 |
| SyncNet         | 18.53              | -                 |
| EEGChannelNet   | 88.09              | -                 |


# How to run

### **Requirements**

- Download dataset
- Place your dataset where you prefer and change the key `dataset_path` on `config.json` file accordingly 
- Create a conda environment through `conda env create -n emer --file environme
nt.yml`
- Optionally, create a wandb account and change the key `wandb_entity` on `config.json` file accordingly. 

All the baselies were tested on a single NVIDIA RTX A6000 GPU.

### **Preprocess data**

```
python preprocess.py --split_dir data/splits
```

### **Train a model for subject identification**

```
python train.py --task subject_identification --model eegnet --lr 0.001 --epochs 100
```

Model weights at best validation accuracy will be saved at exps/subject_identification

### **Train a model for emotion recognition**

```
python train.py --task emotion_recognition --model eegnet --voting_strategy majority --lr 0.001 --epochs 100
```

Model weights at best validation accuracy will be saved at exps/emotion_recognition

### **Inference**

This script will generate te required file for the final submission.
Always specify:
- the task 
- the model architectures
- the path (absolute or relative) to the folder with .pth file

As an example you can run:

```
python inference.py --task subject_identification --model eegnet --voting_strategy mean --resume exps/subject_identification/eegnet/baseline_2024-08-29_16-17-27
```

Running inference on **subject identification** will create a csv file named *results_subject_identification_test_trial.csv* for the held-out-trial test set.

Running inference on **emotion recognition** will create two csv files:
- *results_emotion_recognition_test_trial.csv* for the held-out-trial test set.
- *results_emotion_recognition_test_subject.csv* for the held-out-subject test set.

Each csv has only two columns:
- **id**: the id of the sample
- **prediction**: the predicted class

Here we provide an example of file structure:
```
id,prediction
3784258358,12
1378746257,19
2395445698,8
...
```

# References
[1] V j Lawhern et al., “EEGNet: a compact convolutional neural network for EEG-based brain–computer interfaces”, J. Neural Eng. 15/5, 2018.

[2] Y. Li et al, “Targeting EEG/LFP Synchrony with Neural Nets”, NeurIPS 2017.

[3] S. Palazzo, C. Spampinato, I. Kavasidis, D. Giordano, J. Schmidt and M. Shah, "“Decoding Brain Representations by Multimodal Learning of Neural Activity and Visual Features”, in IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 43, no. 11, 2021.