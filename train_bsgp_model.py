import torch
import numpy as np
from src.datasets.uci_loader import UCIDataset
import seaborn as sns
from src.model_builder import build_model, train
from src.samplers.adaptative_sghmc import AdaptiveSGHMC
import torch.optim as optim
#from torchviz import make_dot
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from src.misc.utils import inf_loop
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
        num_posterior_samples = 100
        posterior_sample_spacing = 32
        window_size  = 64
        mcmc_measures = True
        n_burnin_iters = 8000
        collect_every = 10 # thinning
        K = 10
    args = ARGS()

    bsgp_model = build_model(data_uci.X_train, data_uci.Y_train, args)
    bsgp_model = bsgp_model.to(device)
    bsgp_sampler = AdaptiveSGHMC(bsgp_model.sampling_parameters,
                                lr=args.epsilon, num_burn_in_steps=2000,
                                mdecay=args.mdecay, scale_grad=N)
    bsgp_optimizer = optim.Adam([bsgp_model.likelihood.variance], lr=1e-3)
    n_sampling_iters = args.n_burnin_iters + args.num_posterior_samples * args.collect_every

    if args.mcmc_measures:
        samples_ms_iter = np.empty((data_uci.X_test.shape[0], args.iterations))
        samples_vs_iter = np.empty((data_uci.X_test.shape[0], args.iterations))
        samples_logps_iter = np.empty((data_uci.X_test.shape[0], args.iterations))

    batch_size = args.minibatch_size if N > args.minibatch_size else N
    train_dataset = TensorDataset(torch.Tensor(data_uci.X_train).to(device), torch.Tensor(data_uci.Y_train).to(device))
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True) 

    test_dataset = TensorDataset(torch.Tensor(data_uci.X_test).to(device), torch.Tensor(data_uci.Y_test).to(device))
    test_dataloader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False, drop_last=True) 
    
    iter = 0
    sample_idx = 0
    print(f'Number of iterations: {n_sampling_iters}')
    for iter in range(n_sampling_iters):

        #X_batch = data[0].to(torch.float64) # BCHW
        #Y_batch = data[1].to(torch.float64)

        log_prob = bsgp_model.train_step(device, bsgp_sampler, K=args.K)

        bsgp_model.save_sample('.results/', sample_idx)

        log_prob = bsgp_model.optimizer_step(device, bsgp_optimizer)

        #if (iter > args.n_burnin_iters) and (iter % args.collect_every == 0):
        #    bsgp_model.save_sample('.results/', sample_idx)
        #    sample_idx += 1
        #    bsgp_model.set_samples(SAMPLES_DIR, cache=True)

        if iter % 100 == 0:
            print('TRAIN\t| iter = %6d       sample marginal LL = %5.2f' % (iter, -log_prob.detach()))

            for X, Y in test_dataloader:
                bsgp_model.eval()
                X = X.to(torch.float64)
                Y = Y.to(torch.float64)
                ms, vs = bsgp_model.predict_y(X, len(bsgp_model.window), posterior=False)
                logps = norm.logpdf(np.repeat(Y.numpy()[None, :, :]*Ystd, len(bsgp_model.window), axis=0), ms*Ystd, np.sqrt(vs)*Ystd)
                mnll = -np.mean(logsumexp(logps, axis=0) - np.log(len(bsgp_model.window)))
                print('TEST\t| iter = %6d       MNLL = %5.2f' % (iter, mnll))
        
        iter += 1


if __name__ =='__main__':
    main()