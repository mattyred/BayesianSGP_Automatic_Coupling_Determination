import torch
import wandb
import numpy as np
import os
import json
import argparse
import torch.optim as optim
from src.datasets.uci_loader import UCIDataset
from src.datasets.wcci_loader import WCCIDataset
from src.model_builder import build_model, compute_mnll, compute_accuracy, compute_nrmse
from src.samplers.adaptative_sghmc import AdaptiveSGHMC
from src.misc.utils import ensure_dir, next_path, set_seed

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def save_samples(folder_path, model, **kwargs):
    D = model.D
    K = model.K

    npz_dict = {'D': D, 
                'K': K,
                'WTW': kwargs['WTW'],
                'test_mnll':  kwargs['test_mnll'],
                'test_nrmse': kwargs['test_nrmse']}
    filepath = os.path.join(folder_path, f'vdmgp_data_fold_{kwargs["k"]}')
    np.savez(filepath, **npz_dict)
    if WANDB:
      # Upload to wandb
      kwargs['artifact'].add_file(filepath + '.npz')
      # Log on wandb
      kwargs['run'].log({"test_mnll": kwargs['test_mnll']})


def main(args):
    set_seed(0)
    # Read experiment parameters
    params = vars(args)

    # Configure wandb
    run = None
    run_artifact = None
    global WANDB
    WANDB = params['use_wandb']
    if WANDB:
        # Init Wandb run
        run = wandb.init(
        project="BSGPtorch-wandb",
        name=f"exp_{params['dataset']}_VDMGP_K_{params['num_latents']}",
        config={
            "adam_lr": params['adam_lr'],
            "dataset": params['dataset'],
            "max_iterations": params['max_iterations']
        })
        run_artifact = wandb.Artifact(f"exp_{params['dataset']}_VDMGP_K_{params['num_latents']}_params", type='VDMGP')
    if params['dataset'] not in ['temp', 'so2']:
        data = UCIDataset(dataset=params['dataset'], k=params['kfold'], normalize=True, seed=0)
    else:
        data = WCCIDataset(dataset=params['dataset'], k=params['kfold'], normalize=True, seed=0)

    # Results directory
    run_path = next_path(os.path.dirname('./results/' + '/run-%04d/'))
    kernel_dir = os.path.join(run_path, 'kernel')
    ensure_dir(kernel_dir)
    print(f'Results folder: {run_path}')

    print("GP Model: VDMGP - Titsias 2013")
    print(f"Number of iterations: {params['max_iterations']}")

    for k in range(params['kfold']):
        print(f"\nFOLD {k+1} of {params['kfold']}")
        X_train = data.X_train_kfold[k]
        X_test = data.X_test_kfold[k]
        Y_train = data.Y_train_kfold[k]
        Y_test = data.Y_test_kfold[k]
        Y_train_mean = data.Y_train_mean_kfold[k]
        Y_train_std = data.Y_train_std_kfold[k]
        N = X_train.size(0)
        D = X_train.size(1)

        # Model initialization
        num_latents = min(D, params['num_latents'])
        model = build_model(X_train.to(device), Y_train.to(device), params, num_latents=num_latents, model='VDMGP')
        model = model.to(device)
        print(model)
        optimizer = optim.Adam(model.parameters(), lr=params['adam_lr'])

        # Training
        model.likelihood.variance.requires_grad = False
        num_iterations = params['max_iterations']
        for iter in range(num_iterations):
            if iter > min(num_iterations//2, 2000):
                model.likelihood.variance.requires_grad = True
            elbo = model.train_step(optimizer)
            loss = -elbo.detach()
            if iter % 100 == 0:
                ms, vs = model.predict_y(X_test.to(device))
                test_mnll = compute_mnll(ms, vs, Y_test.numpy(), 1, Y_train_std)
                print('ITER %5d |   Train Loss = %5.2f  |   Test MNLL = %5.2f'%(iter,loss,test_mnll))

        # Save WᵀW samples
        W_samples = model.sample_W(samples=128).detach().numpy()
        WTW = np.matmul(np.transpose(W_samples, axes=(0, 2, 1)), W_samples)
        #WTW = W.T @ W
        #fig, ax = plt.subplots(1, 1, figsize=(3, 3))
        #sns.heatmap(WTW, cmap='vlag', vmin=-np.max(WTW), center=0, vmax=np.max(WTW), square=True, cbar=True, ax=ax)
        #plt.savefig(kernel_dir + '/WTW_mean', dpi=300, bbox_inches='tight', pad_inches=0.2)

        # MNLL performance
        ms, vs = model.predict_y(X_test.to(device))
        test_mnll = compute_mnll(ms, vs, Y_test.numpy(), 1, Y_train_std)
        print('TEST MNLL = %5.2f' % (test_mnll))

        # Task-specific performance
        test_nrmse = compute_nrmse(ms.detach().numpy(), vs, Y_test.numpy(), 1, Y_train_std)
        print('TEST NRMSE = %5.2f' % test_nrmse)

        # Save samples
        save_samples(kernel_dir, model, WTW=WTW, test_mnll=test_mnll, test_nrmse=test_nrmse, artifact=run_artifact, run=run, k=k)

        
    if WANDB:
      run.log_artifact(run_artifact)
      wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='experiment-wandb')
    parser.add_argument('--dataset', type=str, choices=["boston", "kin8nm", "powerplant", "concrete", "breast", "eeg","wilt", "diabetes", "temp", "puma"], default="boston")
    parser.add_argument('--use_wandb', action='store_true')
    parser.add_argument('--num_latents', type=int, default=20)
    parser.add_argument('--num_inducing', type=int, default=10)
    parser.add_argument('--adam_lr', type=float, default=0.01)
    parser.add_argument('--kfold', type=int, default=3)
    parser.add_argument('--max_iterations', type=int, default=20)
    args = parser.parse_args()
    main(args)