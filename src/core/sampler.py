import torch


class SGHMC_sampler(torch.optim.Optimizer):

    def __init__(self, params, mdecay, epsilon):
        super(SGHMC_sampler, self).__init__(params, defaults={'mdecay': mdecay, 'epsilon': epsilon})
        self.mdecay = mdecay
        self.epsilon = epsilon
        self.state = dict() 
        for group in self.param_groups: 
            for p in group['params']: 
                self.state[p] = dict(mom=torch.zeros_like(p.data)) 
        
    
    def step(self, burn_in=True):
        burn_in_updates = []
        sample_updates = []

        #grads = torch.autograd.grad(nll, self.variables, retain_graph=True)

        for theta, grad in zip(self.variables, self.variables.grad):
            xi = torch.nn.Parameter(torch.ones_like(theta), requires_grad=False)
            g = torch.nn.Parameter(torch.ones_like(theta), requires_grad=False)
            g2 = torch.nn.Parameter(torch.ones_like(theta), requires_grad=False)
            p = torch.nn.Parameter(torch.zeros_like(theta), requires_grad=False)

            r_t = 1. / (xi + 1.)
            g_t = (1. - r_t) * g + r_t * grad
            g2_t = (1. - r_t) * g2 + r_t * grad ** 2
            xi_t = 1. + xi * (1. - g * g / (g2 + 1e-16))
            Minv = 1. / (torch.sqrt(g2 + 1e-16) + 1e-16)

            burn_in_updates.extend([(xi, xi_t), (g, g_t), (g2, g2_t)])

            epsilon_scaled = self.epsilon / torch.sqrt(torch.tensor(self.N))
            noise_scale = 2. * epsilon_scaled ** 2 * self.mdecay * Minv
            sigma = torch.sqrt(torch.maximum(noise_scale, torch.tensor(1e-16)))
            sample_t = torch.distributions.normal.Normal(torch.zeros_like(theta), sigma).sample()
            p_t = p - self.epsilon ** 2 * Minv * grad - self.mdecay * p + sample_t
            theta_t = theta + p_t

            sample_updates.extend([(theta, theta_t), (p, p_t)])

        sample_op = sample_updates
        burn_in_op = burn_in_updates + sample_updates

        for group in self.param_groups: 
            for p in group['params']: # bsgp model parameters: U, Z, ker_ls, ker_v
                if p not in self.state: 
                    self.state[p] = dict(mom=torch.zeros_like(p.data)) 
                mom = self.state[p]['mom'] 
                mom = self.momentum * mom - group['lr'] * p.grad.data 
                p.data += mom