# FLAX

# Models
- Inherit from `flax.linen.Module`
- 2 main functions of each module:  **init** and **apply**. 
- Model has **params** and **state**. Params are the model weights. Params is a pytree object (a FrozenDict actually) 

## init
- params = model.init(key, input_shape), the key is the random seed for reproducibility. 

## rngs
- rng = jax.random.PRNGKey(seed)
- key, init_key = jax.random.split(rng)  # the key is a new seed for the next iteration and the init_key is used for the initialization of parameters [init function]

## apply
- output = model.apply(params, input), we pass the params as the state is external. 
- we cannot call the model as model(x), we need to call model.apply(params, x)

# Training
## optax
- optax is a library for gradient based optimization in JAX. Also have init and apply functions.
- opt_state = opt.init(params) -> again the state is handled outside the model
- optax has also learning rate schedulers

## train_state
- train_state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)
- The goal is to keep things together, and do not have to save params, tx_patams, etc. 
