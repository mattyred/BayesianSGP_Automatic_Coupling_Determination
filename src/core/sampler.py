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

        self.sampler_parameters = []
        for theta in list(self.trainable_parameters):
            # sampler parameters (non-trainable)
            xi = torch.nn.Parameter(torch.ones_like(theta), requires_grad=False)
            g = torch.nn.Parameter(torch.ones_like(theta), requires_grad=False)
            g2 = torch.nn.Parameter(torch.ones_like(theta), requires_grad=False)
            p = torch.nn.Parameter(torch.zeros_like(theta), requires_grad=False)
            self.sampler_parameters.append({'xi': xi, 'g': g, 'g2': g2, 'p': p})

    @property
    def trainable_parameters(self):
        return self.param_groups[0]['params']
    
    @property
    def trainable_parameters_gradients(self):
        return [p.grad for p in self.param_groups[0]['params']]
    
    @property
    def non_trainable_parameters(self):
        return self.sampler_parameters
    
    def step(self, burn_in=True):
        burn_in_updates = []
        sample_updates = []

        for theta, grad, sampler_param in zip(self.trainable_parameters, self.trainable_parameters_gradients, self.sampler_parameters):

            r_t = 1. / (sampler_param['xi'] + 1.)
            g_t = (1. - r_t) * sampler_param['g'] + r_t * grad
            g2_t = (1. - r_t) * sampler_param['g2'] + r_t * grad ** 2
            xi_t = 1. + sampler_param['xi'] * (1. - sampler_param['g'] * sampler_param['g'] / (sampler_param['g2'] + 1e-16))
            Minv = 1. / (torch.sqrt(sampler_param['g2'] + 1e-16) + 1e-16)

            burn_in_updates.append((sampler_param['xi'], (xi_t)))
            burn_in_updates.append((sampler_param['g'], (g_t)))
            burn_in_updates.append((sampler_param['g2'], (g2_t)))

            epsilon_scaled = self.epsilon / torch.sqrt(torch.tensor(self.N))
            noise_scale = 2. * epsilon_scaled ** 2 * self.mdecay * Minv
            sigma = torch.sqrt(torch.maximum(noise_scale, torch.tensor(1e-16)))
            sample_t = torch.randn_like(theta) * sigma #torch.distributions.normal.Normal(torch.zeros_like(theta), sigma).sample()
            
            # update trainable parameters and 'p' sampler parameter
            p_t = sampler_param['p'] - self.epsilon ** 2 * Minv * grad - self.mdecay * sampler_param['p'] + sample_t
            theta_t = theta + p_t

            sample_updates.append((theta, theta_t))
            sample_updates.append((sampler_param['p'], p_t))

        if burn_in:
            pass
        else:
            pass