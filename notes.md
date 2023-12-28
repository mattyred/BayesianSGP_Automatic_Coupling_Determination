ToDo
- Add cross validation somehow
- A lengthscale.get() going to 0 causes a crash in the cholesky of Kmm -> gradient clipping? Different PositiveParam?
  - Solution adopted for now: adding a jitter to the lengthscales while doing kernel computations (kernels.py, line 117)

- Adapt kfold runner with new additions
- Adapt BGP model with BSGP with prior, params, ...