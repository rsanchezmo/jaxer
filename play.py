from jaxer.utils.agent import Agent
import jax.numpy as jnp
import time


if __name__ == '__main__':
    agent = Agent(experiment="v0", model_name="2")

    """ LOAD SOME DATA """
    x_test = jnp.ones((1, agent.config.model_config["max_seq_len"], agent.config.model_config["input_features"]))

    """ PREDICT """
    init_t = time.time()
    result = agent(x_test)
    print(f"Time taken: {1e3*(time.time() - init_t):.2f} ms")

    """ GET DATALOADER """
    