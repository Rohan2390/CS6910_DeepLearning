from main import train
import wandb
import json
import argparse

def wandbTrain(config=None):
    with wandb.init(config=config):
        config = wandb.config
        train(config=config,wandbLog=True)

def runSweepId(sweep_id):
    wandb.agent(sweep_id, function=wandbTrain, project="DL_ASSN_2",entity='ed21s001_cs21m030')

def main(wandbConfig):

    sweep_config = {
        'method': 'bayes',
        'metric':{
            'name':'Val Acc',
            'goal':'maximize',
        },
    }

    sweep_config['parameters'] = wandbConfig
    sweep_id = wandb.sweep(sweep_config, project="DL_ASSN_2",entity='ed21s001_cs21m030')
    print(f"Sweep ID:{sweep_id}")
    wandb.agent(sweep_id, function=wandbTrain, project="DL_ASSN_2",entity='ed21s001_cs21m030')


if __name__=='__main__':

    parser = argparse.ArgumentParser(description='Wandb Training')
    parser.add_argument('--path', dest='path', type=str, help='Path to the wandb config json file')
    parser.add_argument('--sweepId', dest='sweepId', type=str, help='Sweep ID of the sweep to run')
    args = parser.parse_args()

    if args.path==None and args.sweepId==None:
        print("Sweep Config path of Config needed.")
    elif args.path and args.sweepId:
        print("Starting new sweep using config given.")
    elif args.path and not args.sweepI:
        path = args.path
        with open(path, 'r') as f:
            wandbConfig = json.load(f)

        main(wandbConfig)
    else:
        runSweepId(args.sweepId)

