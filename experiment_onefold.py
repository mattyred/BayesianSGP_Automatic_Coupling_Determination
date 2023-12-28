import numpy as np
from src.datasets.uci_loader import UCIDataset, DATASET_TASK
from src.model_builder import build_model, compute_mnll, compute_accuracy
from src.samplers.adaptative_sghmc import AdaptiveSGHMC
import torch.optim as optim
#from torchviz import make_dot
from src.misc.utils import inf_loop, ensure_dir, next_path
import os

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
        kernel_cov_data[i,:] = gp_params_dict[param_name].detach()

    npz_dict = {param_name: kernel_cov_data, 
                'D': D_in, 
                'kernel': model.kern.rbf_type,
                'prior': kernel_prior,
                'mnll':  kwargs['mnll']}
    filepath = os.path.join(folder_path, 'kernel_samples')
    np.savez(filepath, **npz_dict)
    return 0


def main():
    dataset_name = 'breast'
    task = DATASET_TASK[dataset_name]
    standardize = task == 'regression'
    data_uci = UCIDataset(dataset_path=f'data/uci/{dataset_name}.pth', k=-1, standardize=standardize, seed=0)

    N, D = data_uci.X_train.shape
    Ystd = data_uci.Y_train_std.numpy()
    print(f'X-train: {N, D}')

    class ARGS():
        num_inducing = 100
        minibatch_size = 1000
        output_dim= 1
        adam_lr = 0.01
        prior_inducing_type = "normal"
        full_cov = False
        epsilon = 0.01
        mdecay = 0.05
        num_posterior_samples = 100
        mcmc_measures = True
        n_burnin_iters = 500
        collect_every = 10 # thinning
        K = 10
        model = 'BSGP'
        kernel_type = 'ACD'
        prior_kernel =  {'type': 'invwishart', 'b': 0.1, 'global_shrinkage': 0.1}
    args = ARGS()

    # Results directyory
    run_path = next_path(os.path.dirname(os.path.realpath(__file__)) + '/results/' + '/run-%04d/')
    samples_dir = os.path.join(run_path, 'samples')
    kernel_dir = os.path.join(run_path, 'kernel')
    ensure_dir(samples_dir)
    ensure_dir(kernel_dir)

    # Model initialization
    model = build_model(data_uci.X_train, data_uci.Y_train, args, model=args.model, task=task)
    model = model.to(device)
    bsgp_sampler = AdaptiveSGHMC(model.sampling_params,
                                lr=args.epsilon, num_burn_in_steps=2000,
                                mdecay=args.mdecay, scale_grad=N)
    if len(model.optimization_params_names) > 0:
        bsgp_optimizer = optim.Adam(model.optim_params, lr=args.adam_lr)
    n_sampling_iters = args.n_burnin_iters + args.num_posterior_samples * args.collect_every
    
    iter = 0
    sample_idx = 0
    #print(f'GP Model: {args.model}')
    print(f'Number of iterations: {n_sampling_iters}')
    print(f'Task: {task}')
    print(model)
    for iter in range(n_sampling_iters):

        log_prob = model.train_step(device, bsgp_sampler, K=args.K)
        if len(model.optimization_params_names) > 0:
            log_prob = model.optimizer_step(device, bsgp_optimizer)

        if (iter > args.n_burnin_iters) and (iter % args.collect_every == 0):
            model.save_sample(samples_dir, sample_idx)
            sample_idx += 1
            model.set_samples(samples_dir, cache=True)

        if iter % 100 == 0:
            print('TRAIN\t| iter = %6d       sample marginal LL = %5.2f' % (iter, -log_prob.detach()))
        iter += 1

    # Measure performance
    model.set_samples(samples_dir, cache=True)
    ms, vs = model.predict_y(data_uci.X_test)
    mnll = compute_mnll(ms, vs, data_uci.Y_test, len(model.gp_samples), Ystd)
    print('TEST MNLL = %5.2f' % (mnll))

    # Task-specific performance
    if task == 'classification':
        accuracy = compute_accuracy(ms, vs, data_uci.Y_test, len(model.gp_samples), Ystd)
        print('TEST Error-Rate = %5.2f' % (1-accuracy))

    # Save posterior samples
    save_samples(kernel_dir, model, mnll=mnll)

if __name__ =='__main__':
    main()