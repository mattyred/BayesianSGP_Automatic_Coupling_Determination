import torch
import numpy as np
from src.datasets.uci_loader import UCIDataset
import seaborn as sns
from src.model_builder import build_model, train
from src.samplers.adaptative_sghmc import AdaptiveSGHMC
from torchviz import make_dot
import matplotlib.pyplot as plt
# setting PyTorch

from src.misc.settings import settings
device = settings.device
if device.type == 'cuda':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')



def main():
    data_uci = UCIDataset(dataset_path='data/uci/boston.pth', static_split=True, seed=0)
    N, D = data_uci.X_train.shape
    print(f'X-train: {N, D}')

    class ARGS():
        num_inducing = 100
        n_layers = 1
        minibatch_size = 100
        window_size = 64
        output_dim= 1
        adam_lr = 0.01
        prior_inducing_type = "uniform"
        full_cov = False
        epsilon = 0.01
        mdecay = 0.05
        iterations = 512
        num_posterior_samples = 100
        posterior_sample_spacing = 32
        mcmc_measures = True
        n_burnin_iters = 500
        collect_every = 10 # thinning
        K = 10
    args = ARGS()

    bsgp_model = build_model(data_uci.X_train, data_uci.Y_train, args)
    bsgp_sampler = AdaptiveSGHMC(bsgp_model.parameters(),
                                lr=args.epsilon, num_burn_in_steps=2000,
                                mdecay=args.mdecay, scale_grad=N)
    n_sampling_iters = args.n_burnin_iters + args.num_posterior_samples * args.collect_every

    if args.mcmc_measures:
        samples_ms_iter = np.empty((data_uci.X_test.shape[0], args.iterations))
        samples_vs_iter = np.empty((data_uci.X_test.shape[0], args.iterations))
        samples_logps_iter = np.empty((data_uci.X_test.shape[0], args.iterations))

    for _ in range(n_sampling_iters):
        nll = train(bsgp_model, bsgp_sampler, args.K)

        if (_ > args.n_burnin_iters) and (_ % args.collect_every == 0):
            bsgp_model.save_sample('.results/', sample_idx)
            sample_idx += 1
            #bsgp_model.set_samples(SAMPLES_DIR, cache=True)

        if _ % 10 == 0:
            print(f'Iter: {_} - Marginal LL: {-nll.detach()}')  


if __name__ =='__main__':
    main()