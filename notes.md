- A lengthscale.get() going to 0 causes a crash in the cholesky of Kmm -> gradient clipping? Different PositiveParam?
  - Solution adopted for now: adding a jitter to the lengthscales while doing kernel computations (kernels.py, line 117)

- Adapt kfold runner with new additions
- Adapt BGP model with BSGP with prior, params, ...

- Understand burnin parameter of AdaptativeSGHMC (?)
- Create .sh files for running
- dump model details as json together with npz results (?)
- Do not print Prior ACD like this, change

## Future (minor) improvments
- Create a common gpmodel class since lots of methods are shared between BSGP and BGP
- Put in some utils file clip_grad function and jacobian function
- Change in kernels (ACD) when using torch.zeros and torch.ones device=... to be device=self.device (thus define a device for the entire kernel), maybe doing kernel.to(device) is enough?
- Change metric computations just to use pytorch