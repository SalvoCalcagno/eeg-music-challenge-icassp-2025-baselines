import os
import argparse
import pandas as pd
from pathlib import Path
import src.trainers as trainers
from src.dataset import get_test_loader

def parse():
    # Init parser
    parser = argparse.ArgumentParser()
    
    # Select task
    parser.add_argument('--task', type=str, default='emotion_recognition', choices=['emotion_recognition', 'subject_identification'])
 
    # Dataset options
    parser.add_argument('--split_dir', type=str, default=str("data/splits"))
    parser.add_argument('--data_type', type=str, default="preprocessed", choices=["raw", "pruned", "preprocessed"])
    parser.add_argument('--preprocessing_pipe', type=str, default="z_score_data/")
    parser.add_argument('--crop_size', type=int, default="1280")
    parser.add_argument('--num_workers', type=int, default=0) 
    
    # Experiment options
    parser.add_argument('-t', '--tag', default='emotion_recognition')
    parser.add_argument('--logdir', default='exps', type=Path)
    
    # Model options
    parser.add_argument('--model', default='eegnet')
    parser.add_argument('--verbose', action='store_true')

    # Mixed model-specific options
    ## EEGNet
    parser.add_argument('--num_channels', type=int, default=32)
    #parser.add_argument('--num_classes', type=int, default=4)
    ## SyncNet
    parser.add_argument('--input_size', type=int, default=1280)#syncnet 276, #rnn 1
    ## EEGChannelNet
    parser.add_argument('--in_channels', type=int, default=1)
    parser.add_argument('--input_width', type=int, default=1280)
    parser.add_argument('--input_height', type=int, default=32)
    parser.add_argument('--num_residual_blocks', type=int, default=2)
    
    # Training options
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--trainer', default='trainer')
    parser.add_argument('--optimizer', default='Adam')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--use_scheduler', action='store_true')
    parser.add_argument('--reduce-lr-every', type=int)
    parser.add_argument('--reduce-lr-factor', type=float, default=0.5)
    parser.add_argument('--patience', type=float, default=10)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--activity-reg-lambda', type=float, help="lambda value for activity regularization")
    parser.add_argument('--momentum', default=0.9, type=float, help="momentum")
    parser.add_argument('--resume')
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, choices=['cuda:0', 'cuda:1', 'cpu'], default='cuda:0')
    parser.add_argument('--multi-gpu', action='store_true')
    parser.add_argument('--overfit-batch', action='store_true')
    parser.add_argument('--voting_strategy', type=str, default='mean', choices=['mean', 'max', 'median', 'min', 'majority'])

    args = parser.parse_args()
    
    # Set number of classes
    if args.task == "emotion_recognition":
        args.num_classes = 4
    elif args.task == "subject_identification":
        args.num_classes = 26
    else:
        raise ValueError("Task not recognized")
    
    args.inference = True
    
    return args

def main(args):

    test_datasets, test_loaders, args = get_test_loader(args)

    # Print dataset info
    for split in test_loaders:
        print(f"{split}: {len(test_loaders[split].dataset)}")

    # Define trainer
    trainer_module = getattr(trainers, args.trainer)
    trainer_class = getattr(trainer_module, 'Trainer')
    trainer = trainer_class(args)

    predictions = trainer.test(test_loaders)
    
    # generate results file
    results = {}
    for split, split_predictions in predictions.items():
        test_dataset = test_datasets[split]
        split_results = []
        for sample, prediction in zip(test_dataset, split_predictions):
            split_results.append({
                'id': sample['id'],
                'prediction': prediction
            })
        results[split] = split_results
    
    for split in results:
        results[split] = pd.DataFrame(results[split])
        results[split].to_csv(f"results_{args.task}_{split}.csv", index=False)

if __name__ == '__main__':
    # Get params
    args = parse()
    main(args)