import json
import wandb
import argparse
from pathlib import Path
import src.config as config
import src.trainers as trainers
from src.dataset import get_loaders

def parse():
    # Init parser
    parser = argparse.ArgumentParser()
    
    wandb_entity = config.get_attribute("wandb_entity")
    
    # Select task
    parser.add_argument('--task', type=str, default='emotion_recognition', choices=['emotion_recognition', 'subject_identification'])
    
    # Enable sweep
    parser.add_argument('--sweep', action='store_true')
    parser.add_argument('--sweep-id', type=str, default=None)
    
    # Dataset options
    parser.add_argument('--split_dir', type=str, default=str("data/splits"))
    parser.add_argument('--data_type', type=str, default="preprocessed", choices=["raw", "pruned", "preprocessed"])
    parser.add_argument('--preprocessing_pipe', type=str, default="z_score_data/")
    parser.add_argument('--crop_size', type=int, default="1280")
    parser.add_argument('--num_workers', type=int, default=0) 
    
    # Experiment options
    parser.add_argument('-t', '--tag', default='emotion_recognition')
    parser.add_argument('--entity', default=f"{wandb_entity}")
    parser.add_argument('--debug', action='store_true') # used for disabling wandb
    parser.add_argument('--logdir', default='exps', type=Path)
    parser.add_argument('--eval-after', type=int, default=-1, help='evaluate only starting from a certain epoch')
    parser.add_argument('--log_every', type=int, default=20)
    parser.add_argument('--save_every', type=int, default=-1)
    parser.add_argument('--watch_model', action='store_true')
    parser.add_argument('--use_voting', action='store_true')
    parser.add_argument('--voting_strategy', type=str, default='mean', choices=['mean', 'max', 'median', 'min', 'majority'])
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
    parser.add_argument('--batch_size', type=int, default=32)
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
    
    args = parser.parse_args()
    
    # Set number of classes
    if args.task == "emotion_recognition":
        args.num_classes = 4
    elif args.task == "subject_identification":
        args.num_classes = 26
    else:
        raise ValueError("Task not recognized")
    
    args.inference = False
    
    return args

def main(args):

    if args.sweep:
        # Load sweep config
        with open("sweep_config.json", "r") as fp:
            sweep_configuration = json.load(fp)
            
        # Initialize sweep
        if args.sweep_id is not None:
            sweep_id = args.sweep_id
        else:
            #sweep_id = wandb.sweep(sweep=sweep_configuration, project=args.task, entity=args.entity)
            sweep_id = "zyjhsm90"

    def train_func():

        if args.sweep:

            # Init run
            run = wandb.init(project=args.task, entity=args.entity)

            # Load parameters
            sweep_params = dict(wandb.config)

            # Update args
            args.__dict__.update(sweep_params)

        loaders, updated_args = get_loaders(args)

        # Print dataset info
        for split in loaders:
            print(f"{split}: {len(loaders[split].dataset)}")

        # Define logdir and tag
        updated_args.logdir = Path(f"exps/{updated_args.task}/{updated_args.model}")
        updated_args.tag = f"baseline"
        
        # Define trainer
        trainer_module = getattr(trainers, updated_args.trainer)
        trainer_class = getattr(trainer_module, 'Trainer')
        trainer = trainer_class(updated_args)

        # Run training
        if updated_args.debug:
            run = wandb.init(project=updated_args.task, entity=updated_args.entity, mode="disabled")
        elif not updated_args.sweep:
            run = wandb.init(project=updated_args.task, entity=updated_args.entity)
        
        # Save Wandb Configuration
        args_dict = vars(updated_args)
        for arg_name, arg in args_dict.items():
            try:
                wandb.config[arg_name] = arg
            except Exception as e:
                print(f"Could not save {arg_name} to wandb")
                continue
        
        run.name = f"{updated_args.task}_{updated_args.model}_{updated_args.tag}"

        
        model, metrics = trainer.train(loaders)

        if not updated_args.sweep:
            # Close saver
            wandb.finish()
        else:
            run.finish()
    
    if args.sweep:
        # Run sweep
        wandb.agent(sweep_id, function=train_func, project=args.task, entity=args.entity)
    else:
        train_func()

if __name__ == '__main__':
    # Get params
    args = parse()
    main(args)