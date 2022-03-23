import argparse

from main import train
import wandb
import json

def wandbTrain(config):
    with wandb.init(config=config):
        config = wandb.config
        train(config, wandbLog=True)

def main(wandnConfig):
    wandb.init(project='DL Assignment 2',entity='ed21s001_cs21m030')

    sweep_config = {
        'method': 'grid',
        'metric':{
            'name':'Val Acc',
            'goal':'maximize',
        },
    }

    sweep_config['parameters'] = wandnConfig
    sweep_id = wandb.sweep(sweep_config, project="DL Assignment 2")
    print(f"Sweep ID:{sweep_id}")
    wandb.agent(sweep_id, wandbTrain)


if __name__=='__main__':

    parser = argparse.ArgumentParser(description='Wandb Training')
    parser.add_argument('--path', dest='path', type=str, help='Path to the wandb config json file')

    with open(parser.parse_args().path, 'r') as f:
        wandbConfig = json.load(f)
