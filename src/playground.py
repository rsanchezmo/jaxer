import jax.numpy as jnp
import numpy as np
import time


def main():
    print("JAX")
    x = jnp.ones((10000, 10000))
    y = jnp.ones((10000, 10000))
    time_init = time.time()
    dot_prod = jnp.dot(x, y)
    time_end = time.time()
    print("JAX time:", time_end - time_init, "seconds")

    print("NumPy")
    x = np.ones((10000, 10000))
    y = np.ones((10000, 10000))
    time_init = time.time()
    dot_prod = np.dot(x, y)
    time_end = time.time()
    print("NumPy time:", time_end - time_init, "seconds")

if __name__ == "__main__":
    main()

