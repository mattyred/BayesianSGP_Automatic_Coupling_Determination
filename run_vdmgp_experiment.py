import torch
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()
import numpy as np
import seaborn as sns
import argparse
from src.datasets.uci_loader import UCIDataset, DATASET_TASK
from src.model_builder import build_model, compute_mnll, compute_accuracy, compute_nrmse
from src.samplers.adaptative_sghmc import AdaptiveSGHMC
import torch.optim as optim
#from torchviz import make_dot
from src.misc.utils import inf_loop, ensure_dir, next_path
import os

from src.misc.settings import settings
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def save_samples(folder_path, model, **kwargs):
    S = len(model.gp_samples)
    D_in = model.kern.input_dim

    if model.kern.rbf_type == 'ACD':
        l_size = model.kern.L.size(0) # D_in*(D_in+1)//2
        kernel_cov_data = np.empty((S, l_size), dtype=np.float64)
        param_name =  'kern.L'
        kernel_prior = model.prior_kernel['type']
    elif model.kern.rbf_type == 'ARD':
        kernel_cov_data = np.empty((S, D_in), dtype=np.float64)
        param_name =  'kern.lengthscales'
        kernel_prior = 'normal'
    else:
        kernel_cov_data = np.empty((S, 1), dtype=np.float64)
        param_name = 'kern.lengthscales'
        kernel_prior = 'normal'

    for i in range(S):
        gp_params_dict = model.gp_samples[i]
        kernel_cov_data[i,:] = gp_params_dict[param_name].cpu().detach()

    npz_dict = {param_name: kernel_cov_data, 
                'D': D_in, 
                'kernel': model.kern.rbf_type,
                'prior': kernel_prior,
                'test_mnll':  kwargs['test_mnll'],
                'test_error_rate': kwargs['test_error_rate'],
                'test_nrmse': kwargs['test_nrmse']}
    filepath = os.path.join(folder_path, 'vdmgp_data')
    np.savez(filepath, **npz_dict)
    return 0


def main():
    dataset_name = args.dataset
    assert dataset_name in DATASET_TASK.keys()
    task = DATASET_TASK[dataset_name]
    data_uci = UCIDataset(dataset=dataset_name, k=-1, normalize=True, seed=0)

    N, D = data_uci.X_train.shape
    Ystd = data_uci.Y_train_std
    print(f'X-train: {N, D}')

    # Results directory
    run_path = next_path(os.path.dirname(os.path.realpath(__file__)) + '/results/' + '/run-%04d/')
    samples_dir = os.path.join(run_path, 'samples')
    kernel_dir = os.path.join(run_path, 'kernel')
    ensure_dir(samples_dir)
    ensure_dir(kernel_dir)
    print(f'Results folder: {run_path}')

    # Model initialization
    model = build_model(data_uci.X_train.to(device), data_uci.Y_train.to(device), vars(args), num_latents=10, model='VDMGP')
    model = model.to(device)

    ## TEST likelihood
    #lik = model.compute_likelihood(data_uci.X_train.to(device), data_uci.Y_train.to(device))
    #mean, var = model.predict_y(data_uci.X_train.to(device))
    optimizer = optim.Adam(model.parameters(), lr=args.adam_lr)
    
    iter = 0
    sample_idx = 0
    print(f'Number of iterations: {args.max_iterations}')
    print(f'Task: {task}')
    #print(f'Gradient clipping: {args.clip_by_value}')
    print(model)
    variance_parameter = model.likelihood.variance
    # First round
    variance_parameter.requires_grad  = False
    for _ in range(2000):
        elbo = model.train_step(optimizer)
        loss = -elbo.detach()
        ms, vs = model.predict_y(data_uci.X_test.to(device))
        test_mnll = compute_mnll(ms, vs, data_uci.Y_test.numpy(), 1, Ystd)
        print('Test MNLL = %5.2f | Train Loss = %5.2f'%(test_mnll, loss))
    # Second round
    variance_parameter.requires_grad = False
    for _ in range(2000):
        model.train_step(optimizer)

    # MNLL performance
    model.set_samples(samples_dir, cache=True)
    ms, vs = model.predict_y(data_uci.X_test.to(device))
    # ms: [num_posterior_samples, Ntest, 1] on column j prediction for data sample j
    test_mnll = compute_mnll(ms, vs, data_uci.Y_test.numpy(), len(model.gp_samples), Ystd)
    print('TEST MNLL = %5.2f' % (test_mnll))

    test_nrmse = compute_nrmse(ms, vs, data_uci.Y_test.numpy(), num_posterior_samples=len(model.gp_samples), ystd=Ystd)
    print('TEST NRMSE = %5.2f' % test_nrmse)

    # Save posterior samples
    save_samples(kernel_dir, model, test_mnll=test_mnll, test_nrmse=test_nrmse)

    writer.flush()
    writer.close()

if __name__ =='__main__':
    parser = argparse.ArgumentParser(description='BSGPtorch - onefold')
    parser.add_argument('--dataset', type=str, default="boston")
    parser.add_argument('--num_inducing', type=int, default=100)
    parser.add_argument('--num_latents', type=int, default=2)
    parser.add_argument('--minibatch_size', type=int, default=1000)
    parser.add_argument('--adam_lr', type=float, default=0.01)
    parser.add_argument('--max_iterations', type=int, default=1500)
    parser.add_argument('--kernel_type', type=str, choices=["ACD", "ARD"], default="ACD")

    args = parser.parse_args()
    main()
