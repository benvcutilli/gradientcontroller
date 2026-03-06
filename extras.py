# Breaking this out into a separate file to simply other files. Not much in here currently,
# but for + in the long term once more and more code gets added to the project as a whole,
# so we want to reduce confusion.

# These packages refer to ^^^^^^ and ^^^^^^, respectively
import jax, clu.metrics

import gradientcontroller

# Equalizes the gradient among pixels as some are seen more than others by parts of the network
###################################################################################################
#                                                                                                 #

def getPixelInfluence(network):

    toStack  = []
    # MNIST^^^^^^ dimensions used here (the 28 and 0)
    for v in range(0, 28):
        for h in range(0, 28):
            image = jax.numpy.zeros((28, 28, 1)).at[v, h, 0].set(1.0)
            toStack.append(image)

    stack    = jax.numpy.stack(toStack, 0)

    counter  = gradientcontroller.PixelInfluenceCalculator(configuration)
    count    = jax.numpy.reshape(
                          counter.init_with_output(KEY, stack)[0][:, 0, 0, 0],
                          (28, 28)
                      )

    return count / count.min()

#weights["constants"] = { "normalizer": normalizedCount }

#                                                                                                  #
####################################################################################################


# ^^^accuracyusage^^^ recommended initializing "results" before the loop and updating it in the
# loop (it makes the most sense this way, so that's why I went with it, but wanted to credit
# them). Accepts any type of network and inputs+. I did this kind of special-purpose funciton in
# ^^^^^^, so using that idea here.
def test(network, weights, images):

    results = clu.metrics.Accuracy.empty()

    # Details about this loop can be found in B at the top of training.py
    for batch in images.iter(50, False):
        prediction = network.apply(weights, batch["image"][:, :, :, None])
        # Setting results to itself is a common code pattern, but I had thought of
        # them doing the same at ^^^reassignment^^^ when figuring out what to do to make this work
        results = results.merge(
            clu.metrics.Accuracy.from_model_output(
                labels=batch["label"],
                logits=prediction
            )
        )

    return results.compute()
