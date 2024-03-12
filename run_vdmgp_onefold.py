import torch
import numpy as np
import seaborn as sns
import argparse
from src.datasets.uci_loader import UCIDataset
from src.datasets.wcci_loader import WCCIDataset
from src.model_builder import build_model, compute_mnll, compute_nrmse
import torch.optim as optim
from src.misc.utils import inf_loop, ensure_dir, next_path
import os
import matplotlib.pyplot as plt
from src.misc.settings import settings
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def save_samples(folder_path, model, **kwargs):
    D = model.D
    K = model.K

    npz_dict = {'D': D, 
                'K': K,
                'WTW': kwargs['WTW'],
                'test_mnll':  kwargs['test_mnll'],
                'test_nrmse': kwargs['test_nrmse']}
    filepath = os.path.join(folder_path, 'vdmgp_data')
    np.savez(filepath, **npz_dict)
    return 0


def main():
    #data_uci = WCCIDataset(dataset='temp', k=-1, normalize=True, seed=0)
    data_uci = UCIDataset(dataset=args.dataset, k=-1, normalize=True, seed=0)

    N, D = data_uci.X_train.shape
    Ystd = data_uci.Y_train_std
    print(f'X-train: {N, D}')

    # Results directory
    run_path = next_path(os.path.dirname(os.path.realpath(__file__)) + '/results/' + '/run-%04d/')
    kernel_dir = os.path.join(run_path, 'kernel')
    ensure_dir(kernel_dir)
    print(f'Results folder: {run_path}')

    # Model initialization
    num_latents = min(D, args.num_latents)
    model = build_model(data_uci.X_train.to(device), data_uci.Y_train.to(device), vars(args), num_latents=num_latents, model='VDMGP')
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.adam_lr)
    
    print(f'Number of iterations: {args.max_iterations}')
    print(model)

    # Training
    model.likelihood.variance.requires_grad  = False
    for i in range(args.max_iterations):
        if i > args.max_iterations//2:
            model.likelihood.variance.requires_grad = True
        elbo = model.train_step(optimizer)
        loss = -elbo.detach()
        if i % 100 == 0:
            ms, vs = model.predict_y(data_uci.X_test.to(device))
            test_mnll = compute_mnll(ms, vs, data_uci.Y_test.numpy(), 1, Ystd)
            print('ITER %5d |   Train Loss = %5.2f  |   Test MNLL = %5.2f'%(i,loss,test_mnll))

    # Save WᵀW samples
    W = model.sample_W(samples=128).detach().numpy().mean(axis=0)
    WTW = W.T @ W
    fig, ax = plt.subplots(1, 1, figsize=(3, 3))
    sns.heatmap(WTW, cmap='vlag', vmin=-np.max(WTW), center=0, vmax=np.max(WTW), square=True, cbar=True, ax=ax)
    plt.savefig(kernel_dir + '/WTW_mean', dpi=300, bbox_inches='tight', pad_inches=0.2)

    # MNLL performance
    ms, vs = model.predict_y(data_uci.X_test.to(device))
    # ms: [num_posterior_samples, Ntest, 1] on column j prediction for data sample j
    test_mnll = compute_mnll(ms, vs, data_uci.Y_test.numpy(), 1, Ystd)
    print('TEST MNLL = %5.2f' % (test_mnll))

    test_nrmse = compute_nrmse(ms.detach().numpy(), vs, data_uci.Y_test.numpy(), 1, Ystd)
    print('TEST NRMSE = %5.2f' % test_nrmse)

    # Save posterior samples
    save_samples(kernel_dir, model, WTW=WTW, test_mnll=test_mnll, test_nrmse=test_nrmse)

if __name__ =='__main__':
    parser = argparse.ArgumentParser(description='VDMGPtorch - onefold')
    parser.add_argument('--dataset', type=str, default="boston")
    parser.add_argument('--num_inducing', type=int, default=10)
    parser.add_argument('--num_latents', type=int, default=10)
    parser.add_argument('--adam_lr', type=float, default=0.01)
    parser.add_argument('--max_iterations', type=int, default=4000)

    args = parser.parse_args()
    main()
