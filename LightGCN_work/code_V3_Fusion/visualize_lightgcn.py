import os
import pandas as pd
import matplotlib.pyplot as plt
import torch
from model import LightGCN
from data import DataLoader  # adjust import to your loader

def plot_training_curves(log_dir):
    train_csv = os.path.join(log_dir, 'train_epoch_metrics.csv')
    valid_csv = os.path.join(log_dir, 'valid_epoch_metrics.csv')
    if not os.path.exists(train_csv) or not os.path.exists(valid_csv):
        print(f"Metrics not found in {log_dir}")
        return
    train_df = pd.read_csv(train_csv)
    valid_df = pd.read_csv(valid_csv)
    for col in train_df.columns:
        if col == 'epoch': continue
        plt.figure()
        plt.plot(train_df['epoch'], train_df[col], label=f'train_{col}')
        if col in valid_df:
            plt.plot(valid_df['epoch'], valid_df[col], label=f'valid_{col}')
        plt.title(col)
        plt.xlabel('Epoch')
        plt.legend()
        plt.grid()
        plt.show()

def plot_alpha_distribution(model_path, data_path, device='cpu'):
    # load data & model
    data = DataLoader(data_path)
    config = {'n_layers': data.config['n_layers']}
    model = LightGCN(config, data).to(device)
    ckpt = torch.load(model_path, map_location=device)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    with torch.no_grad():
        _ = model.computer()
        # reuse same degree-based alpha logic
        u_deg = data.interactions.user_deg
        i_deg = data.interactions.item_deg
        import numpy as np
        deg = np.concatenate([u_deg, i_deg]).astype(float)
        alpha = torch.from_numpy(deg/deg.sum()).to(device)
        alpha = alpha.view(-1,1).repeat(1, model.n_layers+1)
        alpha = alpha / alpha.sum(dim=1, keepdim=True)
    alpha = alpha.cpu().numpy()
    # histogram per layer
    for k in range(alpha.shape[1]):
        plt.figure()
        plt.hist(alpha[:,k], bins=50)
        plt.title(f'Layer {k} alpha')
        plt.xlabel('Weight')
        plt.ylabel('Frequency')
        plt.grid()
        plt.show()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', required=True)
    parser.add_argument('--model_path')
    parser.add_argument('--data_path')
    args = parser.parse_args()
    plot_training_curves(args.log_dir)
    if args.model_path and args.data_path:
        plot_alpha_distribution(args.model_path, args.data_path)
