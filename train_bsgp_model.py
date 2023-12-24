import torch
import numpy as np
from src.datasets.uci_loader import UCIDataset
import seaborn as sns
from src.model_builder import build_bsgp_model, build_bgp_model, compute_mnll
from src.samplers.adaptative_sghmc import AdaptiveSGHMC
import torch.optim as optim
#from torchviz import make_dot
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from src.misc.utils import inf_loop, ensure_dir, next_path
import os
from scipy.stats import norm
from scipy.special import logsumexp

from src.misc.settings import settings
device = 'cpu' #'cuda' if torch.cuda.is_available() else 'cpu'



def main():
    data_uci = UCIDataset(dataset_path='data/uci/boston.pth', static_split=True, seed=0)
    N, D = data_uci.X_train.shape
    Ystd = data_uci.Y_train_std.numpy()
    print(f'X-train: {N, D}')

    class ARGS():
        num_inducing = 100
        n_layers = 1
        minibatch_size = 1000
        window_size = 64
        output_dim= 1
        adam_lr = 0.01
        prior_inducing_type = "normal"
        full_cov = False
        epsilon = 0.01
        mdecay = 0.05
        iterations = 512
        num_posterior_samples = 256
        posterior_sample_spacing = 32
        window_size  = 64
        mcmc_measures = True
        n_burnin_iters = 500
        collect_every = 10 # thinning
        K = 10
        model = 'BGP'
    args = ARGS()

    # Results directyory
    run_path = next_path(os.path.dirname(os.path.realpath(__file__)) + '/results/' + '/run-%04d/')
    samples_dir = os.path.join(run_path, 'samples')
    ensure_dir(samples_dir)

    # Model initialization
    if args.model == 'BGP':
        model = build_bgp_model(data_uci.X_train, data_uci.Y_train, args)
    elif args.model == 'BSGP':
        model = build_bsgp_model(data_uci.X_train, data_uci.Y_train, args)
    model = model.to(device)
    bsgp_sampler = AdaptiveSGHMC(model.sampling_params,
                                lr=args.epsilon, num_burn_in_steps=2000,
                                mdecay=args.mdecay, scale_grad=N)
    bsgp_optimizer = optim.Adam([model.optim_params], lr=args.adam_lr)
    n_sampling_iters = args.n_burnin_iters + args.num_posterior_samples * args.collect_every

    if args.mcmc_measures:
        samples_ms_iter = np.empty((data_uci.X_test.shape[0], args.iterations))
        samples_vs_iter = np.empty((data_uci.X_test.shape[0], args.iterations))
        samples_logps_iter = np.empty((data_uci.X_test.shape[0], args.iterations))
    
    iter = 0
    sample_idx = 0
    print(f'GP Model: {args.model}')
    print(f'Number of iterations: {n_sampling_iters}')
    for iter in range(n_sampling_iters):

        log_prob = model.train_step(device, bsgp_sampler, K=args.K)
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

    # Save posterior samples

if __name__ =='__main__':
    main()