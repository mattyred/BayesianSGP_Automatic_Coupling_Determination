import torch


class SGHMC_sampler(torch.optim.Optimizer):

    def __init__(self, params, N, mdecay, epsilon):
        super(SGHMC_sampler, self).__init__(params, defaults={'mdecay': mdecay, 'epsilon': epsilon})
        self.mdecay = mdecay
        self.epsilon = epsilon
        self.N = N
        for group in self.param_groups: 
            for p in group['params']: 
                self.state[p] = dict(mom=torch.zeros_like(p.data)) 

    @property
    def trainable_parameters(self):
        return self.param_groups[0]['params']
    
    @property
    def trainable_parameters_gradients(self):
        return [p.grad for p in self.param_groups[0]['params']]
    
    def step(self, sampler_parameters, burn_in=True):
        for theta, grad, theta_sampler_params in zip(self.trainable_parameters, self.trainable_parameters_gradients, sampler_parameters):
            r_t = 1. / (theta_sampler_params['xi'] + 1.)
            g_t = (1. - r_t) * theta_sampler_params['g'] + r_t * grad
            g2_t = (1. - r_t) * theta_sampler_params['g2'] + r_t * grad ** 2
            xi_t = 1. + theta_sampler_params['xi'] * (1. - theta_sampler_params['g'] * theta_sampler_params['g'] / (theta_sampler_params['g2'] + 1e-16))
            Minv = 1. / (torch.sqrt(theta_sampler_params['g2'] + 1e-16) + 1e-16)

            if burn_in:
                theta_sampler_params['xi'].data = xi_t
                theta_sampler_params['g'].data = g_t
                theta_sampler_params['g2'].data = g2_t

            epsilon_scaled = self.epsilon / torch.sqrt(torch.tensor(self.N))
            noise_scale = 2. * epsilon_scaled ** 2 * self.mdecay * Minv
            sigma = torch.sqrt(torch.maximum(noise_scale, torch.tensor(1e-16)))
            sample_t = torch.distributions.normal.Normal(torch.zeros_like(theta), sigma).sample()
            
            # update trainable parameters and 'p' sampler parameter
            p_t = theta_sampler_params['p'] - self.epsilon ** 2 * Minv * grad - self.mdecay * theta_sampler_params['p'] + sample_t
            theta_t = theta + p_t

            theta_sampler_params['p'].data = p_t
            theta.data = theta_t # trainable parameters update
