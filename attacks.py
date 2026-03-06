# This file does the same thing as attacks.py^^^^^^ (and has the same name). It is convenient
# to have things in separate files, and each attack is also still a function. The file also
# allows separation of church and state ("separation of concerns"), which aids the goals of +.

# Using ^^^^^^ framework, as well as ^^^^^^. Also importing ^^^^^^.
import jax, optax, flax.training.common_utils

# Implementing FGSM^^^^^^
def fastGradient(data, network, weights, magnitude):

    # Need a function I can pass to jax.grad; this is it
    def evaluate(network, weights, data, target):
        # Same loss function code (see "penalty") used in training.py, so any comments there are
        # relevant here
        return jax.numpy.sum(
                   optax.squared_error(
                       network.apply(weights, data),
                       target
                   )
               )

    target = jax.numpy.expand_dims(
                 flax.training.common_utils.onehot( data["label"],  10 ),
                 0
             )

    pixels = jax.numpy.expand_dims(
                 data["image"].astype(jax.numpy.float32),
                 0
             )
    
    backwards  = jax.grad(evaluate, 2)
    corruption = magnitude * jax.numpy.sign(
                                 backwards(
                                     network,
                                     weights,
                                     pixels,
                                     target
                                 )[0]
                             )

    return { "image": pixels + corruption }
