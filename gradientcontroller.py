# May use later:
# Taking a page from ^^^representationlearning^^^, an
# extra five layers to the network (between
# the fourth and the last) because convergence was an issue.

# A. The third parameter to CompoundLayer in all but the first layer avoided being
#    hard-coded so that we can be flexible+



# Because one of the goals of this code is to create an alternative test bench to Cleverhans^^^^^^,
# anything that participates in this has a "+" put next to it.

# JAX and Flax are developed by ^^^^^^, ^^^^^^. Flax was recommended by ^^^^^^.
####################################################################################################
#                                                                                                   #

import jax
import flax

#                                                                                                   #
####################################################################################################


TARGET = 1e2

debug = None

# The user can change+ the settings of the convolutional/linear layers used here by passing a
# sequence of them (in dictionary format) in as allConvolutionalMetas/allLinearMetas to the
# constructor.
####################################################################################################
#                                                                                                  #

# Fully convolutional just like in ^^^representationlearning^^^. See CompoundLayer for
# rationale for the standardLayerOnly attribute+.
class ConvolutionalNetwork(flax.linen.Module):

    allConvolutionalMetas: dict
    standardLayerOnly: bool

    def setup(self):

        # A discusses part of what is going on here
        self.layer1 = CompoundLayer(
                          GradientController(self.allConvolutionalMetas[0],
                                             TARGET),
                          self.allConvolutionalMetas[0],
                          # MNIST^^^^^^ has one layer of input
                          1,
                          self.standardLayerOnly
                      )
        self.layer2 = CompoundLayer(
                          GradientController(self.allConvolutionalMetas[1],
                                             TARGET),
                          self.allConvolutionalMetas[1],
                          self.allConvolutionalMetas[0]["features"],
                          self.standardLayerOnly
                      )
        self.layer3 = CompoundLayer(
                          GradientController(self.allConvolutionalMetas[2],
                                             TARGET),
                          self.allConvolutionalMetas[2],
                          self.allConvolutionalMetas[1]["features"],
                          self.standardLayerOnly
                      )
        self.layer4 = CompoundLayer(
                          GradientController(self.allConvolutionalMetas[3],
                                             TARGET),
                          self.allConvolutionalMetas[3],
                          self.allConvolutionalMetas[2]["features"],
                          self.standardLayerOnly
                      )
        self.layer5 = CompoundLayer(
                          GradientController(self.allConvolutionalMetas[4],
                                             TARGET),
                          self.allConvolutionalMetas[4],
                          self.allConvolutionalMetas[3]["features"],
                          self.standardLayerOnly
                      )
        self.layer6 = CompoundLayer(
                          GradientController(self.allConvolutionalMetas[5],
                                             TARGET),
                          self.allConvolutionalMetas[5],
                          self.allConvolutionalMetas[4]["features"],
                          self.standardLayerOnly
                      )
        self.layer7 = CompoundLayer(
                          GradientController(self.allConvolutionalMetas[6],
                                             TARGET),
                          self.allConvolutionalMetas[6],
                          self.allConvolutionalMetas[5]["features"],
                          self.standardLayerOnly
                      )


    def __call__(self, inputs):

        #inputs = inputs / jax.numpy.expand_dims(
        #                      self.scope.get_variable("constants", "normalizer", inputs),
        #                      2
        #                  )

        # I write these networks often, so I am re-using my strategy of assigning a variable
        # to each part of the network at every step, passing it to the next
        ############################################################################################
        #                                                                                           #

        layer1Out = self.layer1(inputs)
        layer1Act = flax.linen.activation.relu(layer1Out)

        layer2Out = self.layer2(layer1Act)
        layer2Act = flax.linen.activation.relu(layer2Out)

        layer3Out = self.layer3(layer2Act)
        layer3Act = flax.linen.activation.relu(layer3Out)

        layer4Out = self.layer4(layer3Act)
        layer4Act = flax.linen.activation.relu(layer4Out)

        layer5Out = self.layer5(layer4Act)
        layer5Act = flax.linen.activation.relu(layer5Out)

        layer6Out = self.layer6(layer5Act)
        layer6Act = flax.linen.activation.relu(layer6Out)

        layer7Out = self.layer7(layer6Act)
        layer7Act = flax.linen.softmax(layer7Out) 

        #                                                                                           #
        ############################################################################################

        return jax.numpy.squeeze(layer7Act)



# Only mirrors the convolutional part of ConvolutionalNetwork, no activations or anything else
class PixelInfluenceCalculator(flax.linen.Module):

    allConvolutionalMetas: dict

    def setup(self):

        self.layer1 = flax.linen.Conv(**self.allConvolutionalMetas[0],
                                      bias_init=jax.nn.initializers.zeros,
                                      kernel_init=jax.nn.initializers.ones)
        self.layer2 = flax.linen.Conv(**self.allConvolutionalMetas[1],
                                      bias_init=jax.nn.initializers.zeros,
                                      kernel_init=jax.nn.initializers.ones)
        self.layer3 = flax.linen.Conv(**self.allConvolutionalMetas[2],
                                      bias_init=jax.nn.initializers.zeros,
                                      kernel_init=jax.nn.initializers.ones)
        self.layer4 = flax.linen.Conv(**self.allConvolutionalMetas[3],
                                      bias_init=jax.nn.initializers.zeros,
                                      kernel_init=jax.nn.initializers.ones)
        self.layer5 = flax.linen.Conv(**self.allConvolutionalMetas[4],
                                      bias_init=jax.nn.initializers.zeros,
                                      kernel_init=jax.nn.initializers.ones)
        self.layer6 = flax.linen.Conv(**self.allConvolutionalMetas[5],
                                      bias_init=jax.nn.initializers.zeros,
                                      kernel_init=jax.nn.initializers.ones)
        self.layer7 = flax.linen.Conv(**self.allConvolutionalMetas[6],
                                      bias_init=jax.nn.initializers.zeros,
                                      kernel_init=jax.nn.initializers.ones)

    def __call__(self, inputs):

        layer1Out = self.layer1(inputs)
        layer2Out = self.layer2(layer1Out)
        layer3Out = self.layer3(layer2Out)
        layer4Out = self.layer4(layer3Out)
        layer5Out = self.layer5(layer4Out)
        layer6Out = self.layer6(layer5Out)
        layer7Out = self.layer7(layer6Out)

        return layer7Out









class LinearNetwork(flax.linen.Module):

    allLinearMetas: dict

    def setup(self):

        self.layer1 = CompoundLayer(
                          GradientController(
                              TARGET
                          ),
                          flax.linen.Dense(
                              **self.allLinearMetas[0],
                              #kernel_init=jax.nn.initializers.ones,
                              bias_init=jax.nn.initializers.zeros,
                              use_bias=True
                          ),
                          784, # Number of pixels in an MNIST image^^^^^^
                          False
                      )
        self.layer2 = CompoundLayer(
                          GradientController(
                              TARGET
                          ),
                          flax.linen.Dense(
                              **self.allLinearMetas[1],
                              #kernel_init=jax.nn.initializers.ones,
                              bias_init=jax.nn.initializers.zeros,
                              use_bias=False
                          ),
                          self.allLinearMetas[0]["features"], # An explanation for this line is in A
                          False
                      )

    def __call__(self, inputs):

        inputs = inputs.reshape(inputs.shape[0], -1)

        layer1Out = self.layer1(inputs)
        layer1Act = flax.linen.activation.sigmoid(layer1Out) 
        layer2Out = self.layer2(layer1Act)
        layer2Act = flax.linen.softmax(layer2Out)
        
        global debug
        debug = jax.numpy.count_nonzero(jax.numpy.abs(layer1Out) > 15)

        return layer2Act

#                                                                                                  #
####################################################################################################




























@jax.custom_jvp
def slider(value):

    return jax.nn.sigmoid(value[0])

@slider.defjvp
def slider_derivative(value, multiplyBy):

    return slider(value), 1 * multiplyBy[0]


# Same rationale for self.metas as allConvolutionalMetas above, but it is
# just one element from that array. In the same spirit, users can also specify an instance "partner"
# to be "partnered" with it.
class CompoundLayer(flax.linen.Module):

    partner:           flax.linen.Module
    main:              flax.linen.Module
    # See where this attribute is used in setup(...) below for related comment
    previousDepth:     int
    # Disables/enables "partner" in __call__
    standardLayerOnly: bool

    def setup(self):

        # GradientController needs to have initialized kernels from the layer right after it, so
        # in order to do that, this "dummy" array forces the layer to do so in that layer's __call__
        # method, an idea credited to init_with_output^^^^^^ (as I think it probably does the same)
        ############################################################################################
        #                                                                                          #

        dummy = None
        if isinstance(self.main, flax.linen.Conv):
            dummy = jax.numpy.zeros((1, 500, 500, self.previousDepth))
        else:
            dummy = jax.numpy.zeros((1, self.previousDepth))

        self.main(dummy)

        #                                                                                          #
        ############################################################################################


    def __call__(self, activations):

        partnerOutput       = self.partner(activations)

        # This section's inline conditionals allow us to pass the input directly to convolution so
        # that we can have a plain network. It is useful for benchmarking+ as well as isolating
        # parts. The motivation for this was that the feature was present in ^^^^^^.
        ############################################################################################
        #                                                                                          #

        # Rogger^^^localization^^^ does dictionary merges, which inspired this (but the code isn't
        # copied from there). Combining the whole dictionary hopefully increases this line's
        # generality as far as whether it works for just flax.linen.Conv or something else+.
        parameters          = (self.main.scope.variables().copy() | partnerOutput[1])              \
                                      if not self.standardLayerOnly else                           \
                                      self.main.scope.variables().copy()

        # Writing a string of attribute/method calls here I think may have been inspired by
        # other code; I might have specifically had an image in my head of the "apply" being after
        # something, as in I had seen it somewhere else, but I don't know how true that is
        mainOutput = self.main.apply(
                         parameters,
                         activations if self.standardLayerOnly else partnerOutput[0]
                     )

        #                                                                                          #
        ############################################################################################

        return mainOutput



# For use as the "partner" attribute in CompoundLayer. Purposely avoids passing in
# information about self.main (we don't want to be too specific with our code+). This
# reduced intermingling is accomplished via scope manipulation (at the time of this writing, any
# code involving self.scope).
class GradientController(flax.linen.Module):

    raiseFrom: float

    def setup(self):

        #fractionHiddensSize      = self.scope.parent.push("main", reuse=True)             \
        #                                            .get_variable("params", "kernel")              \
        #                                            .shape


        #self.fractionHiddens     = self.param("fractionHiddens",
        #                                      flax.linen.initializers.ones,
        #                                      fractionHiddensSize,
        #                                      jax.numpy.float32)

        #self.fractionAdherences  = self.param("fractionAdherences",
        #                                      flax.linen.initializers.zeros,
        #                                      fractionHiddensSize,
        #                                      jax.numpy.float32)

        self.layerHidden         = self.param("layerHidden",
                                              flax.linen.initializers.ones,
                                              [],
                                              jax.numpy.float32)

        fractionHiddensSize      = self.scope.parent.push("main", reuse=True)             \
                                                    .get_variable("params", "kernel")              \
                                                    .shape[:-1] + (1,)

        self.fractionHiddens     = self.param("fractionHiddens",
                                              flax.linen.initializers.zeros,
                                              fractionHiddensSize,
                                              jax.numpy.float32)


    def __call__(self, values):

        extractor       = flax.traverse_util.ModelParamTraversal(
                              lambda path, variable: path.endswith("layerHidden")
                          )

        allLayerHiddens = list( extractor.iterate(self.scope.parent.parent.variables()) )

        # I think maybe it was thesis^^^representationlearning^^^ that gave me the idea to call
        # *.abs() for the absolute value or to use absolute value in the first place (even though it
        # is part of this layer's mathematical formulation anyway)
        bottomGradient  = jax.numpy.sum(
                              jax.numpy.abs(
                                  # Reminded (although I'm not 100% sure I knew hstack specifically
                                  # existed; maybe one of its cousins like vstack was what I used
                                  # and/or seen previously in other framework(s)) that I can use this
                                  # function to put all these values together by a resource I can't
                                  # remember
                                  jax.numpy.hstack(allLayerHiddens)
                              )
                          )
        topGradient      = jax.numpy.abs(self.layerHidden)

        # Same comments here as can be found for bottomGradient
        bottomWeights = jax.numpy.sum(
                            jax.numpy.abs(
                                jax.numpy.hstack(
                                  self.scope.parent.push("main", reuse=True)                       \
                                  .get_variable("params", "kernel")
                                )
                            )
                        )

        topWeights    = self.scope.parent.push("main", reuse=True).get_variable("params", "kernel")
                                


        #utilizedFractions = (self.fractionHiddens / jax.numpy.sum(self.fractionHiddens))           \
        #                                              *                                            \
        #                            jax.nn.sigmoid(self.fractionAdherences)

        #weightScale       = (self.raiseFrom ** (topGradient / bottomGradient))                     \
        #                                         *                                                 \
        #                                 utilizedFractions

        weightScale       = (self.raiseFrom ** (topGradient / bottomGradient))                     \
                                                   *                                               \
                            flax.linen.activation.sigmoid(self.fractionHiddens)

        normalizedWeights =  topWeights                                                            \
                                 /                                                                 \
                            bottomWeights

        result            = normalizedWeights * weightScale

        # The first thing in this tuple is essentially a no-op because this layer doesn't ever
        # mathematically touch it, but it does need to be passed in to its the other layer in
        # CompoundLayer; this design facilitates the self.partner-agnostic design in
        # CompoundLayer.
        return values, {
                           "params": {
                               "kernel": result,
                               "bias": self.scope.parent.push("main", reuse=True)                  \
                                       .get_variable("params", "bias")
                           }
                       }

