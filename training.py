# (A) I didn't know how to use jax.random.PRNGKey with these methods (as a seed), but
# ^^^keyconversion^^^ reminded me that I can convert it to something acceptable with one of
# JAX's sampling methods (in this case, bits(...)).
#
# (B) ^^^sectionfromgrainfrom^^^ gave me the idea of using a for loop directly on the iterator (instead of indexing),
# plus just using a for loop in general). This was originally written to be used with Grain, but
# had to switch to Datasets instead. Only thing that changed was the iterable used in the for
# loop in addition to batch now being indexed into with strings like "image" and "label"
# (although doing that and with strings might have been necessary to do with Grain anyway).

import gradientcontroller
# ^^^grainfrom^^^ suggested using Grain^^^^^^
#import grain

# Huggingface Datasets^^^^^^ was found using Google
import datasets

# optax^^^^^^ via ^^^^^^
import optax

# Package from ^^^^^^
import jax

# Package can be found at ^^^^^^
import flax, flax.training.common_utils

# Making ^^^^^^ available
import clu.metrics

# ^^^pythonpickle^^^ needed
import pickle


import extras

KEY = jax.random.PRNGKey(86939)


# Smaller validation chosen because that is what we did in ^^^representationlearning^^^. Loading
# MNIST^^^^^^.
data = datasets.load_dataset("mnist")["train"]                                                     \
               .train_test_split(0.08,
                                 shuffle=True,
                                 # See comment A
                                 seed=jax.random.bits(KEY).item())

KEY = jax.random.split(KEY)[1]


data_optimization = data["train"]
data_performance  = data["test"]
data_optimization.set_format("jax")
data_performance.set_format("jax")

# These use stride 2 where noted because I did the same in ^^^representationlearningnetwork^^^.
# In an effort to make this objective learnable (I take this mentality from somewhere in
# ^^^representationlearning^^^), layers have more neurons.
configuration  = ( 
     flax.core.FrozenDict({
        "precision": "high",
        "padding": 1,
        "kernel_size": (3, 3),
        "features": 30,
        "strides": 1
     }), # 28 x 28
     flax.core.FrozenDict({
        "precision": "high",
        "padding": ((0, 1), (0, 1)),
        "kernel_size": (3, 3),
        "features": 30,
        "strides": 2
     }), # 13 x 13
     flax.core.FrozenDict({
        "precision": "high",
        "padding": 1,
        "kernel_size": (3, 3),
        "features": 30,
        "strides": 1
     }), # 13 x 13
     flax.core.FrozenDict({
        "precision": "high",
        "padding": 0,
        "kernel_size": (3, 3),
        "features": 30,
        "strides": 2
     }), # 6 x 6
     flax.core.FrozenDict({
        "precision": "high",
        "padding": ((0, 1), (0, 1)),
        "kernel_size": (3, 3),
        "features": 30,
        "strides": 2
     }), # 3 x 3
     flax.core.FrozenDict({
        "precision": "high",
        "padding": 0,
        "kernel_size": (2, 2),
        "features": 30,
        "strides": 1
     }), # 2 x 2
     flax.core.FrozenDict({
        "precision": "high",
        "padding": ((0, 1), (0, 1)),
        "kernel_size": (3, 3),
        # This value is to accommodate the outputs required for MNIST^^^^^^
        "features": 10
     })
)

configuration = (
    flax.core.FrozenDict({
        "features": 400
    }),
    # "features" is set for MNIST^^^^^^ output
    flax.core.FrozenDict({
        "features": 10
    })
)

# Doing squared error here most likely because that is what I had used before in
# ^^^representationlearning^^^ or some other previous work (well, I think it was some form of
# squared error, maybe mean squared error or root mean squared error)
def penalty(target, network, parameters, batch):
    return jax.numpy.sum(
               optax.squared_error(
                   network.apply(parameters,
                       			 batch),
                   target
               )
           )
derivative = jax.jit(
                jax.value_and_grad(penalty, argnums=(2, 3)),
                static_argnums=1
             )
#derivative = jax.value_and_grad(penalty, argnums=(2, 3))

model          = gradientcontroller.ConvolutionalNetwork(configuration, False)
model          = gradientcontroller.LinearNetwork(configuration)

# I think I had seen somewhere before that the parameters need to be declared and initialized
# outside the loop because we can't do any object-oriented stuff with JAX. Actually we could
# probably initialize it inside the loop, but I had seen it this way instead and I like it better.
# Also, the idea of initializing by sending in one of the images in the dataset might be from
# somewhere else too.
weights    = model.init(KEY, data_optimization[0]["image"][None, :, :, None])


optimizer      = optax.sgd(0.00023, 0.999)#optax.rmsprop(0.008)
optimizerState = optimizer.init(weights)



# Didn't used to need an outer loop for epochs when I attempted to use Grain, but since we aren't
# anymore, it is needed
epochs = 80
for epoch in range(epochs):

    # Grain would automatically shuffle this for us, so I needed to add this line in when I switched
    # away from it. Also, comment A is relevant here. I didn't set data_optimization to itself (but
    # shuffled), but I ended up doing it at least in part, if not in whole, because that's what
    # ^^^reassignment^^^ did.
    data_optimization = data_optimization.shuffle(jax.random.bits(KEY).item())
    KEY = jax.random.split(KEY)[1]

    loss = -1
    jacobianAccumulator = jax.numpy.zeros(())
    # Comment B is relevant here
    for batch in data_optimization.iter(50, True):

        inputTensor     = batch["image"][:, :, :, None].astype(jax.numpy.float32)
        truth           = flax.training.common_utils.onehot( batch["label"],  10 )

        loss, jacobian                  = derivative(truth, model, weights, inputTensor) 
        modifications, optimizerState   = optimizer.update(jacobian[0], optimizerState, weights)
        # We make sure (in order to be generic+) any type of constants stay constant via reassigning
        # this after apply_updates below ----------------------------------------------------
        #constants                       = weights["constants"]                            # |
        weights                         = optax.apply_updates(weights, modifications)     # |
        #weights["constants"]            = constants # <--------------------------------------

        jacobianAccumulator += jax.numpy.abs(jacobian[1]).sum( (1, 2, 3) ).sum()

        #print(jacobian[0]["params"]["layer1"]["main"]["kernel"])
        #import time
        #time.sleep(10)
    
    jacobianAccumulator /= 1164
    print(gradientcontroller.debug)#model.apply(weights, batch["image"][:, :, :, None].astype(jax.numpy.float32) / 255))
    a = jacobian[0]["params"]["layer1"]["main"]["kernel"]
    b = jacobian[0]["params"]["layer1"]["main"]["bias"]
    c = jacobian[0]["params"]["layer2"]["main"]["kernel"]
    #d = jacobian[0]["params"]["layer2"]["main"]["bias"]
    e = jacobian[0]["params"]["layer1"]["partner"]["layerHidden"]
    f = jacobian[0]["params"]["layer2"]["partner"]["layerHidden"]
    print(f"layer1: weights - {a.var():.2e} around {a.mean():.2e}; bias - {b.var():.2e} around {b.mean():.2e};  {e:.2e}")
    print(f"layer2: weights - {c.var():.2e} around {c.mean():.2e}; {f:.2e}")
    #print(jax.numpy.sum(jax.numpy.abs(jacobian[1])) / 50)
    #import time
    #time.sleep(10)

    print(loss)


    # Validation
    ################################################################################################
    #                                                                                              #

    validationPerformance = extras.test(model, weights, data_performance)
    print("validation for epoch {0}: {1}".format(epoch, validationPerformance))

    #                                                                                              #
    ################################################################################################

everything = {
    "weights": weights,
    # The inspiration for this line is that, in ^^^^^^, I saved training settings to a file
    "forConstructor": configuration,
}
handle = open("theta", "wb")
pickle.dump(everything, handle, pickle.HIGHEST_PROTOCOL)
handle.close()
