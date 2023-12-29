ToDo
- A lengthscale.get() going to 0 causes a crash in the cholesky of Kmm -> gradient clipping? Different PositiveParam?
  - Solution adopted for now: adding a jitter to the lengthscales while doing kernel computations (kernels.py, line 117)

- Adapt kfold runner with new additions
- Adapt BGP model with BSGP with prior, params, ...
- Change in kernels (ACD) when using torch.zeros and torch.ones device=... to be device=self.device (thus define a device for the entire kernel)
- Update pytorch on Mac (version 2.1.2)
- Understand burnin parameter of AdaptativeSGHMC
- Save task-specific performances in .npz files (nrmse and accuracy/error-rate)
- Add nrmse computation
- Create .sh files for running