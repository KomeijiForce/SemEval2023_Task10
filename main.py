import argparse
import json
import pandas as pd
import numpy as np
import torch

from models import SexismDetector

seed=514
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True

def run_experiment(config_path, data_path):
    config = json.load(open(config_path))
    df_all = pd.read_csv(data_path)
    #df_all = pd.read_csv('/vault/lepeng/jupyter/SemEval2023/SemEval2023/edos_labelled_aggregated.csv')
    df_train = df_all[df_all.split=='train']
    df_dev = df_all[df_all.split=='dev']
    df_test = df_all[df_all.split=='test']

    if config['disable_neg_train']:
        df_train = df_train[df_train.label_sexist != 'not sexist']
        df_train = df_train.sample(frac=1)

    if config['disable_neg_eval']:
        df_dev = df_dev[df_dev.label_sexist != 'not sexist']
        df_test = df_test[df_test.label_sexist != 'not sexist']
        df_dev = df_dev.sample(frac=1)
        df_test = df_test.sample(frac=1)

    sexism_detector = SexismDetector(config)

    best_dev = 0.0
    best_ep = 0
    early_stop = 0
    for epoch in range(config["epochs"]):
        print(f"#Epoch {epoch}")
        sexism_detector.train(df_train, config)
        s_dev = sexism_detector.evaluate(df_dev, target=config['target'], dataset="Dev")
        if s_dev > best_dev:
            best_dev = s_dev
            best_ep = epoch
            s_test = sexism_detector.evaluate(df_test, target=config['target'], dataset="Test")
            early_stop = 0
            if config["save_file"]:
                savefile = config["path"].split("/")[-1] + "_model.bin"
                torch.save(sexism_detector.state_dict(), savefile)
        else:
            early_stop += 1
        
        if early_stop == 3:
            print(f"\n\t*** Early stop at epoch {epoch}!!\n")
            break
            
    print("Best ep: ", best_ep, best_dev, s_test)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_file", type=str, default=None, help="A json file containing the configuration."
    )
    parser.add_argument(
        "--data_file", type=str, default=None, help="A csv containing the whole(train/dev/test) data."
    )
    args = parser.parse_args()
    run_experiment(args.config_file, args.data_file)
