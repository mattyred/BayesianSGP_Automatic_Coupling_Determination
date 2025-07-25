import torch
import numpy as np
import os
import json
import argparse
import torch.optim as optim
from src.datasets.uci_loader import UCIDataset, DATASET_TASK
from src.model_builder import build_model, compute_mnll, compute_accuracy, compute_nrmse
from src.samplers.adaptative_sghmc import AdaptiveSGHMC
from src.misc.utils import ensure_dir, next_path, set_seed
#from torch.utils.tensorboard import SummaryWriter
#writer = SummaryWriter()

set_seed(0)
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
                'Pd': kwargs['Pd'],
                'prior': kernel_prior,
                'test_mnll':  kwargs['test_mnll'],
                'test_error_rate': kwargs['test_error_rate'],
                'test_nrmse': kwargs['test_nrmse'],
                'test_predictions': kwargs['test_predictions']}
    
    # Store validation metrics
    for metric_key in kwargs['validation_metrics_dict'].keys():
        metric_iter = kwargs['validation_metrics_dict'][metric_key]
        if len(metric_iter) > 0:
            npz_dict[metric_key] = np.array(metric_iter)

    # Save locally
    filepath = os.path.join(folder_path, f"data_chain_{kwargs['chain']}")
    np.savez(filepath, **npz_dict)


def main(args):
    # Read experiment parameters
    #os.chdir('./BSGPtorch') for lightining.ai
    params_folder = './experiments'
    with open(os.path.join(params_folder,'defaults.json'), 'r') as file:
      default_params = json.load(file)
    with open(os.path.join(params_folder, args.experiment + '.json'), 'r') as file:
      exp_params = json.load(file)
    default_params.update(exp_params)
    params = default_params
    params['model'] = args.model
    params['dataset'] = args.dataset
    params['num_inducing'] = args.num_inducing
    params['pca_latents'] = args.pca_latents

    run = None
    run_artifact = None

    dataset_name = params['dataset']
    assert dataset_name in DATASET_TASK.keys()
    task = DATASET_TASK[dataset_name]
    data_uci = UCIDataset(dataset=dataset_name, k=-1, load_static_split=True, normalize=True, pca_latents=params['pca_latents'], seed=0)

    # ACD prior args
    prior_kernel = None
    if params['kernel_type'] == 'ACD':
        prior_kernel =  {
        'kernel': 'ACD',
        'type': params['prior_kernel_type'], 
        'b': params['b'], 
        'global_shrinkage': params['global_shrinkage'],
        'm': params['m'],
        'v': params['v']}
    else:
        prior_kernel =  {'kernel': 'ARD'}

    # Results directory
    run_path = next_path(os.path.dirname('./results/' + '/run-%04d/'))
    samples_dir = os.path.join(run_path, 'samples')
    kernel_dir = os.path.join(run_path, 'kernel')
    ensure_dir(samples_dir)
    ensure_dir(kernel_dir)
    print(f'Results folder: {run_path}')

    n_sampling_iters = params['n_burnin_iters'] + params['num_posterior_samples'] * params['collect_every']
    clip_value = 1 / (params['epsilon'] * 10) if params['clip_by_value'] else None # s.t. clip_value <= 1 / epsilon
    print(f"GP Model: {params['model']}")
    print(f"Number of iterations (each chain): {n_sampling_iters}")
    print(f"Gradient clipping: {params['clip_by_value']}")
    print(f"Data split: 80% train / 20% test [pre-loaded]")

    X_train = data_uci.X_train
    X_test = data_uci.X_test
    Y_train = data_uci.Y_train
    Y_test = data_uci.Y_test
    Y_train_mean = data_uci.Y_train_mean
    Y_train_std = data_uci.Y_train_std
    N = X_train.size(0)

    test_nrmse_iter = []
    test_error_rate_iter = []
    test_mnll_iter = []
    # test_predictions = []
    ll_iter = []
    iter = 0
    sample_idx = 0
    num_chains = 4

    # Run MCMC chains
    for chain in range(num_chains):
        # Initilize (reset) model
        test_mnll_iter = []
        model = build_model(X_train.to(device), Y_train.to(device), params, model=params['model'], prior_kernel=prior_kernel, task=task)
        print(model) if chain == 0 else None
        bsgp_sampler = AdaptiveSGHMC(model.sampling_params,
                                lr=params['epsilon'], num_burn_in_steps=2000,
                                mdecay=params['mdecay'], scale_grad=N)
        if len(model.optimization_params_names) > 0:
            bsgp_optimizer = optim.Adam(model.optim_params, lr=params['adam_lr'])

        # Samples folder
        fold_samples_dir = os.path.join(samples_dir, f'samples_chain_{chain}')
        ensure_dir(fold_samples_dir)

        # Training loop
        print(f'\n##### MCMC CHAIN {chain+1}/{num_chains} #####')
        for iter in range(n_sampling_iters):

            log_prob = model.train_step(device, bsgp_sampler, K=params['K'], clip_value=clip_value)
            if (len(model.optimization_params_names) > 0):
                log_prob = model.optimizer_step(device, bsgp_optimizer, clip_value=100)

            if (iter > params['n_burnin_iters']) and (iter % params['collect_every'] == 0):
                model.save_sample(fold_samples_dir, sample_idx)
                sample_idx += 1
                model.set_samples(fold_samples_dir, cache=True)

            if iter % 100 == 0:
                ll = -log_prob.detach()
                print('TRAIN\t| iter = %6d\t training loss =\t %5.2f' % (iter, ll))
                ll_iter.append(ll.item())

            # Validation
            if iter % 50 == 0:
                with torch.no_grad():
                    y_mean, y_var = model.predict(X_test.to(device))
                    ms, vs = np.stack([y_mean.cpu().detach()], 0), np.stack([y_var.cpu().detach()], 0)
                    #writer.add_scalar(f"chain_{chain+1}/train/lik-variance", model.likelihood.variance.get().numpy(), iter)
                    #writer.add_scalar(f"chain_{chain+1}/train/Train LL", ll.item(), iter)
                    #writer.add_scalar(f"chain_{chain+1}/test/pred-post-sample-0", ms[0,0], iter)
                    #writer.add_scalar(f"chain_{chain+1}/test/pred-post-sample-2", ms[0,2], iter)
                    test_mnll = compute_mnll(ms, vs, Y_test.numpy(), 1, Y_train_std, task=task)
                    test_mnll_iter.append(test_mnll)
                    #writer.add_scalar(f"chain_{chain+1}/test/Test MNLL", test_mnll, iter)
                    if task == 'classification':
                        accuracy = compute_accuracy(ms, vs, Y_test.numpy(), 1, Y_train_std)
                        test_error_rate = 1 - accuracy
                        test_error_rate_iter.append(test_error_rate)
                    elif task == 'regression':
                        test_nrmse = compute_nrmse(ms, vs, Y_test.numpy(), num_posterior_samples=1, ystd=Y_train_std)
                        test_nrmse_iter.append(test_nrmse)

        # Test
        model.set_samples(fold_samples_dir, cache=True)
        ms, vs = model.predict_y(X_test.to(device))

        # Print chain's performance
        test_mnll = compute_mnll(ms, vs, Y_test.numpy(), len(model.gp_samples), Y_train_std, task=task)
        print('\nTEST MNLL =\t %5.2f' % (test_mnll))

        test_nrmse = None
        test_error_rate = None
        if task == 'classification':
            accuracy = compute_accuracy(ms, vs, Y_test.numpy(), len(model.gp_samples), Y_train_std)
            test_error_rate = 1 - accuracy
            print('TEST Error-Rate =\t %.2f%%' % (test_error_rate*100))
        elif task == 'regression':
            test_nrmse = compute_nrmse(ms, vs, Y_test.numpy(), num_posterior_samples=len(model.gp_samples), ystd=Y_train_std)
            print('TEST NRMSE =\t %5.2f' % test_nrmse)

        # Save samples
        validation_metrics_dict = {'test_nrmse_iter': test_nrmse_iter, 
                                    'test_error_rate_iter': test_error_rate_iter, 
                                    'test_mnll_iter': test_mnll_iter, 
                                    'll_iter': ll_iter}
        save_samples(kernel_dir, model, chain=chain,
                        artifact=run_artifact, run=run,
                        test_mnll=test_mnll,
                        test_error_rate=test_error_rate,
                        test_nrmse=test_nrmse,
                        test_predictions=ms,
                        validation_metrics_dict=validation_metrics_dict,
                        Pd = data_uci.Pd)
        #writer.flush()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='experiment-wandb')
    parser.add_argument('--experiment', type=str, default="")
    parser.add_argument('--model', type=str, choices=["BSGP", "BGP"], default="BSGP")
    parser.add_argument('--dataset', type=str, choices=["boston", "kin8nm", "powerplant", "concrete", "breast", "eeg", "wilt", "diabetes", "puma"], default="boston")
    parser.add_argument('--use_wandb', action='store_true')
    parser.add_argument('--num_inducing', type=int, default=500)
    parser.add_argument('--pca_latents', type=int, default=-1)
    args = parser.parse_args()
    main(args)