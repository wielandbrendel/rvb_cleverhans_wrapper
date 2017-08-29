from cleverhans.model import Model
from tensorflow.python.framework import ops
import tensorflow as tf
import numpy as np

class RobustVisionModel(Model):

    """
    An interface to apply CleverHans attacks in the Robust Vision Benchmark.
    """

    def __init__(self, adversarial):
        """
        :param adversarial: interface object provided by the robust vision benchmark
        """
        self.adversarial = adversarial
        self.layer_names = ['logits']

    def _logits_op(self, x, name=None):
        print('HERE', type(x))

        def _logits_gradient(op, grad):
            x = op.inputs[0]
            gradient = lambda x: self.adversarial.gradient(x, loss='logits')
            return gradient(x)

        with ops.name_scope(name, "logits", [x]) as name:
            logits = lambda x: self.adversarial.batch_predictions(x, strict=False)
            op = self._py_func(logits,
                                [x],
                                [tf.float32],
                                name=name,
                                grad=_logits_gradient)
        
        return op[0]

    @staticmethod
    def _py_func(func, inp, Tout, stateful=True, name=None, grad=None):
        """
        Define custom py_func which takes also a grad op as argument.
        """
        # Need to generate a unique name to avoid duplicates:
        rnd_name = 'PyFuncGrad' + str(np.random.randint(0, 1E+8))

        tf.RegisterGradient(rnd_name)(grad)  # see _MySquareGrad for grad example
        g = tf.get_default_graph()
        with g.gradient_override_map({"PyFunc": rnd_name, "PyFuncStateless": rnd_name}):
            return tf.py_func(func, inp, Tout, stateful=stateful, name=name)

    def get_layer(self, x, layer):
        """
        Expose the hidden features of a model given a layer name.
        :param x: A symbolic representation of the network input
        :param layer: The name of the hidden layer to return features at.
        :return: A symbolic representation of the hidden features
        :raise: NoSuchLayerError if `layer` is not in the model.
        """
        # Return the symbolic representation for this layer.
        output = self.fprop(x)
        try:
            requested = output[layer]
        except KeyError:
            raise NoSuchLayerError()
        return requested

    def get_logits(self, x):
        """
        :param x: A symbolic representation of the network input
        :return: A symbolic representation of the output logits (i.e., the
                 values fed as inputs to the softmax layer).
        """
        return self.get_layer(x, 'logits')

    def get_probs(self, x):
        """
        :param x: A symbolic representation of the network input
        :return: A symbolic representation of the output probabilities (i.e.,
                the output values produced by the softmax layer).
        """
        return self.get_layer(x, 'probs')

    def fprop(self, x):
        """
        Exposes all the layers of the model returned by get_layer_names.
        :param x: A symbolic representation of the network input
        :return: A dictionary mapping layer names to the symbolic
                 representation of their output.
        """
        return {'probs' : self._logits_op(x)}
