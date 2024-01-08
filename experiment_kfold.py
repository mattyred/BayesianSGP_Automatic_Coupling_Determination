import torch
import numpy as np
from src.datasets.uci_loader import UCIDataset, DATASET_TASK
import seaborn as sns
from src.model_builder import build_model, compute_mnll, compute_accuracy, compute_nrmse
from src.samplers.adaptative_sghmc import AdaptiveSGHMC
import torch.optim as optim
from src.misc.utils import ensure_dir, next_path
import os
from scipy.stats import norm
import argparse

from src.misc.settings import settings
device = 'cpu' #'cuda' if torch.cuda.is_available() else 'cpu'

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
    filepath = os.path.join(folder_path, 'kernel_samples')
    np.savez(filepath, **npz_dict)
    return 0

def main():
    dataset_name = 'boston'
    task = DATASET_TASK[dataset_name]
    standardize = task == 'regression'
    if task == 'classification':
        assert args.model == 'BSGP' 
    data_uci = UCIDataset(dataset_path=f'data/uci/{dataset_name}.pth', k=args.kfold, standardize=standardize, seed=0)

    # ACD prior args
    prior_kernel = None
    if args.kernel_type == 'ACD':
        prior_kernel =  {'type': args.prior_kernel_type, 'b': args.b, 'global_shrinkage': args.global_shrinkage}

    # Results directory
    run_path = next_path(os.path.dirname(os.path.realpath(__file__)) + '/results/' + '/run-%04d/')
    samples_dir = os.path.join(run_path, 'samples')
    kernel_dir = os.path.join(run_path, 'kernel')
    ensure_dir(samples_dir)
    ensure_dir(kernel_dir)
    print(f'Results folder: {run_path}')

    n_sampling_iters = args.n_burnin_iters + args.num_posterior_samples * args.collect_every
    clip_value = 1 / (args.epsilon * 10) if args.clip_by_value else None # s.t. clip_value <= 1 / epsilon
    print(f'GP Model: {args.model}')
    print(f'Number of iterations: {n_sampling_iters}')
    print(f'Gradient clipping: {args.clip_by_value}')

    for k in range(args.kfold):
        print(f'\nFOLD {k+1} of {args.kfold}')
        X_train = data_uci.X_train_kfold[k]
        X_test = data_uci.X_test_kfold[k]
        Y_train = data_uci.Y_train_kfold[k]
        Y_test = data_uci.Y_test_kfold[k]
        Y_train_mean = data_uci.Y_train_mean_kfold[k]
        Y_train_std = data_uci.Y_train_std_kfold[k]
        N = X_train.size(0)

        # Folder initialization
        fold_samples_dir = os.path.join(samples_dir, f'fold_{k}')
        ensure_dir(fold_samples_dir)

        # Model initialization
        model = build_model(X_train.to(device), Y_train.to(device), args, model=args.model, prior_kernel=prior_kernel, task=task)
        print(model)
        model = model.to(device)
        bsgp_sampler = AdaptiveSGHMC(model.sampling_params,
                                    lr=args.epsilon, num_burn_in_steps=2000,
                                    mdecay=args.mdecay, scale_grad=N)
        if len(model.optimization_params_names) > 0:
            bsgp_optimizer = optim.Adam(model.optim_params, lr=args.adam_lr)

        iter = 0
        sample_idx = 0
        for iter in range(n_sampling_iters):

            log_prob = model.train_step(device, bsgp_sampler, K=args.K, clip_value=clip_value)
            if len(model.optimization_params_names) > 0:
                log_prob = model.optimizer_step(device, bsgp_optimizer, clip_value=clip_value)

            if (iter > args.n_burnin_iters) and (iter % args.collect_every == 0):
                model.save_sample(fold_samples_dir, sample_idx)
                sample_idx += 1
                model.set_samples(fold_samples_dir, cache=True)

            if iter % 100 == 0:
                print('TRAIN\t| iter = %6d       sample marginal LL =\t %5.2f' % (iter, -log_prob.detach()))
            iter += 1

        # MNLL performance
        model.set_samples(fold_samples_dir, cache=True)
        ms, vs = model.predict_y(X_test.to(device))
        # ms: [num_posterior_samples, Ntest, 1] on column j prediction for data sample j
        test_mnll = compute_mnll(ms, vs, Y_test.numpy(), len(model.gp_samples), Y_train_std)
        print('\nTEST MNLL =\t %5.2f' % (test_mnll))

        # Task-specific performance
        test_nrmse = None
        test_error_rate = None
        if task == 'classification':
            accuracy = compute_accuracy(ms, vs, Y_test.numpy(), len(model.gp_samples), Y_train_std)
            test_error_rate = 1 - accuracy
            print('TEST Error-Rate =\t %.2f%%' % (test_error_rate*100))
        elif task == 'regression':
            test_nrmse = compute_nrmse(ms, vs, Y_test.numpy(), num_posterior_samples=len(model.gp_samples), ystd=Y_train_std)
            print('TEST NRMSE =\t %5.2f' % test_nrmse)

        # Save posterior samples
        save_samples(kernel_dir, model, test_mnll=test_mnll, test_error_rate=test_error_rate, test_nrmse=test_nrmse)

if __name__ =='__main__':
    parser = argparse.ArgumentParser(description='BSGPtorch - onefold')
    parser.add_argument('--num_inducing', type=int, default=100)
    parser.add_argument('--minibatch_size', type=int, default=1000)
    parser.add_argument('--adam_lr', type=float, default=0.01)
    parser.add_argument('--prior_inducing_type', type=str, choices=["normal", "uniform", "strauss"], default="normal")
    parser.add_argument('--full_cov', type=bool, default=False)
    parser.add_argument('--epsilon', type=float, default=0.01) 
    parser.add_argument('--clip_by_value', action='store_true')
    parser.add_argument('--mdecay', type=float, default=0.05)
    parser.add_argument('--num_posterior_samples', type=int, default=100)
    parser.add_argument('--n_burnin_iters', type=int, default=1500)
    parser.add_argument('--collect_every', type=int, default=50)
    parser.add_argument('--K', type=int, default=10)
    parser.add_argument('--model', type=str, choices=["BSGP", "BGP"], default="BSGP")
    parser.add_argument('--kernel_type', type=str, choices=["ARD", "ACD"], default="ACD")
    parser.add_argument('--prior_kernel_type', type=str, choices=["wishart", "invwishart", "laplace", "horseshoe", "normal"], default="wishart")
    parser.add_argument('--b', type=float, default=0.1)
    parser.add_argument('--global_shrinkage', type=float, default=0.1)
    parser.add_argument('--kfold', type=int, default=3)
    args = parser.parse_args()
    main()