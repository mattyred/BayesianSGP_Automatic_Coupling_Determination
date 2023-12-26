import torch
import numpy as np
from src.datasets.uci_loader import UCIDataset
import seaborn as sns
from src.model_builder import build_bsgp_model, build_bgp_model, compute_mnll
from src.samplers.adaptative_sghmc import AdaptiveSGHMC
import torch.optim as optim
from src.misc.utils import ensure_dir, next_path
import os
from scipy.stats import norm

from src.misc.settings import settings
device = 'cpu' #'cuda' if torch.cuda.is_available() else 'cpu'

def save_samples(folder_path, model, kfold=0):
    S = len(model.gp_samples)
    D_in = model.kern.input_dim

    if model.kern.rbf_type == 'ACD':
        l_size = model.kern.L.size(0) # D_in*(D_in+1)//2
        kernel_cov_data = np.empty((S, l_size), dtype=np.float64)
        param_name =  'kern.L'
    elif model.kern.rbf_type == 'ARD':
        kernel_cov_data = np.empty((S, D_in), dtype=np.float64)
        param_name =  'kern.lengthscales'
    else:
        kernel_cov_data = np.empty((S, 1), dtype=np.float64)
        param_name = 'kern.lengthscales'

    for i in range(S):
        gp_params_dict = model.gp_samples[i]
        kernel_cov_data[i,:] = gp_params_dict[param_name].detach()

    filepath = os.path.join(folder_path, f'kernel_samples_fold_{kfold}')
    np.savez(filepath, param_name=kernel_cov_data)
    return 0


def main():
    class ARGS():
        num_inducing = 100
        minibatch_size = 1000
        output_dim= 1
        adam_lr = 0.01
        prior_inducing_type = "normal"
        full_cov = False
        epsilon = 0.01
        mdecay = 0.05
        num_posterior_samples = 10
        mcmc_measures = True
        n_burnin_iters = 10
        collect_every = 10 # thinning
        K = 10
        model = 'BSGP'
        kernel_type = 'ARD'
        prior_kernel =  {'type': 'laplace', 'b': 0.1}
        kfold = 5
    args = ARGS()
    
    #  Load data
    data_uci = UCIDataset(dataset_path='data/uci/boston.pth', k=args.kfold, seed=0)

    # Results directyory
    run_path = next_path(os.path.dirname(os.path.realpath(__file__)) + '/results/' + '/run-%04d/')
    samples_dir = os.path.join(run_path, 'samples')
    kernel_dir = os.path.join(run_path, 'kernel')
    ensure_dir(samples_dir)
    ensure_dir(kernel_dir)

    n_sampling_iters = args.n_burnin_iters + args.num_posterior_samples * args.collect_every
    print(f'GP Model: {args.model}')
    print(f'Number of iterations: {n_sampling_iters}')

    for k in range(args.kfold):
        print(f'FOLD {k} of {args.kfold}')
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
        if args.model == 'BGP':
            model = build_bgp_model(X_train, Y_train, args)
        elif args.model == 'BSGP':
            model = build_bsgp_model(X_train, Y_train, args)
        model = model.to(device)
        bsgp_sampler = AdaptiveSGHMC(model.sampling_params,
                                    lr=args.epsilon, num_burn_in_steps=2000,
                                    mdecay=args.mdecay, scale_grad=N)
        bsgp_optimizer = optim.Adam([model.optim_params], lr=args.adam_lr)
        
        iter = 0
        sample_idx = 0
        for iter in range(n_sampling_iters):

            log_prob = model.train_step(device, bsgp_sampler, K=args.K)
            log_prob = model.optimizer_step(device, bsgp_optimizer)

            if (iter > args.n_burnin_iters) and (iter % args.collect_every == 0):
                model.save_sample(fold_samples_dir, sample_idx)
                sample_idx += 1
                model.set_samples(fold_samples_dir, cache=True)

            if iter % 100 == 0:
                print('\tTRAIN\t| iter = %6d       sample marginal LL = %5.2f' % (iter, -log_prob.detach()))
            iter += 1

        # Measure performance
        model.set_samples(fold_samples_dir, cache=True)
        ms, vs = model.predict_y(X_test)
        mnll = compute_mnll(ms, vs, Y_test, len(model.gp_samples), Y_train_std)
        print('\tTEST MNLL = %5.2f' % (mnll))

        # Save posterior samples
        save_samples(kernel_dir, model, kfold=k)

if __name__ =='__main__':
    main()