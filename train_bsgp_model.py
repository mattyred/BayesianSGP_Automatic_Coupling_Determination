import torch
import numpy as np
from src.datasets.uci_loader import UCIDataset
import seaborn as sns
from src.model_builder import build_model, train
from src.samplers.adaptative_sghmc import AdaptiveSGHMC
#from torchviz import make_dot
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from src.misc.utils import inf_loop
# setting PyTorch

from src.misc.settings import settings
device = 'cuda' if torch.cuda.is_available() else 'cpu'



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
    bsgp_model = bsgp_model.to(device)
    bsgp_sampler = AdaptiveSGHMC(bsgp_model.parameters(),
                                lr=args.epsilon, num_burn_in_steps=2000,
                                mdecay=args.mdecay, scale_grad=N)
    n_sampling_iters = args.n_burnin_iters + args.num_posterior_samples * args.collect_every

    if args.mcmc_measures:
        samples_ms_iter = np.empty((data_uci.X_test.shape[0], args.iterations))
        samples_vs_iter = np.empty((data_uci.X_test.shape[0], args.iterations))
        samples_logps_iter = np.empty((data_uci.X_test.shape[0], args.iterations))

    train_dataset = TensorDataset(torch.Tensor(data_uci.X_train).to(device), torch.Tensor(data_uci.Y_train).to(device))
    train_dataloader = DataLoader(train_dataset, batch_size=args.minibatch_size, shuffle=True, drop_last=True) 
    
    iter = 0
    print(f'Number of iterations: {n_sampling_iters}')
    for data in inf_loop(train_dataloader):
        if iter > n_sampling_iters:
            break

        X_batch = data[0].to(torch.float64) # BCHW
        Y_batch = data[1].to(torch.float64)

        #nll = train(bsgp_model, bsgp_sampler, args.K)
        for k in range(args.K):
            #X_batch, Y_batch = bsgp_model.get_minibatch()
            log_prob = bsgp_model.log_prob(X_batch, Y_batch)
            bsgp_sampler.zero_grad()
            loss = -log_prob
            loss.backward()
            bsgp_sampler.step()

        if (iter > args.n_burnin_iters) and (iter % args.collect_every == 0):
            bsgp_model.save_sample('.results/', sample_idx)
            sample_idx += 1
            #bsgp_model.set_samples(SAMPLES_DIR, cache=True)

        if iter % 50 == 0:
            print(f'Iter: {iter} - Marginal LL: {log_prob.detach()}')  

        iter += 1


if __name__ =='__main__':
    main()