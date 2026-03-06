# Importing the packages "datasets"^^^^^^, "flax"^^^^^^, "jax"^^^^^^, argparse^^^^^^, pathlib^^^^^^,
# and pickle^^^^^^
import datasets
import flax
import jax
import argparse
import pathlib
import pickle

import attacks
import gradientcontroller
import extras


commandLine = argparse.ArgumentParser()
commandLine.add_argument("magnitude", type=float)
commandLine.add_argument("trainingOutput", type=pathlib.Path)

settings = commandLine.parse_args()

# Loading selected+ network, dataset, and attack
####################################################################################################
#                                                                                                  #

images = datasets.load_dataset("mnist")["test"]
images.set_format("jax")

handle    = open(settings.trainingOutput, "rb")
everything   = pickle.load(handle)
handle.close()

weights   = everything["weights"]
network   = gradientcontroller.LinearNetwork(everything["forConstructor"])

remake    = attacks.fastGradient


#                                                                                                  #
####################################################################################################

# Modifying images
####################################################################################################
#                                                                                                  #

images = images.map(
    remake,
    fn_kwargs={
                  "network": network,
                  "weights": weights,
                  "magnitude": settings.magnitude
              },
    keep_in_memory=True
)

#                                                                                                  #
####################################################################################################

# Just as in ^^^^^^, we simply call an auxiliary function that is shared across the codebase that
# does performance calculations.
result = extras.test(network, weights, images)
print(f"{result * 100}% correct")
