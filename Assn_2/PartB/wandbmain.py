from main import train
import wandb
import json
import argparse

def wandbTrain(config=None):
    with wandb.init(config=config):
        config = wandb.config
        train(config=config,wandbLog=True)

def main(wandbConfig):
    wandb.init(project='DL Assignment 2',entity='ed21s001_cs21m030')

    sweep_config = {
        'method': 'grid',
        'metric':{
            'name':'Val Acc',
            'goal':'maximize',
        },
    }

    sweep_config['parameters'] = wandbConfig
    sweep_id = wandb.sweep(sweep_config, project="DL Assignment 2")
    print(f"Sweep ID:{sweep_id}")
    wandb.agent(sweep_id, wandbTrain)


if __name__=='__main__':

    parser = argparse.ArgumentParser(description='Wandb Training')
    parser.add_argument('--path', dest='path', type=str, help='Path to the wandb config json file')
    args = parser.parse_args()

    if args.path==None:
        path = 'config.json'
    else:
        path = args.path

    with open(path, 'r') as f:
        wandbConfig = json.load(f)

    main(wandbConfig)
