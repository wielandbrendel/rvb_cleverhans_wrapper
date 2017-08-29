import tensorflow as tf
from tensorflow.python.framework import ops
import numpy as np

# Define custom py_func which takes also a grad op as argument:
def py_func(func, inp, Tout, stateful=True, name=None, grad=None):

    # Need to generate a unique name to avoid duplicates:
    rnd_name = 'PyFuncGrad' + str(np.random.randint(0, 1E+8))

    tf.RegisterGradient(rnd_name)(grad)  # see _MySquareGrad for grad example
    g = tf.get_default_graph()
    with g.gradient_override_map({"PyFunc": rnd_name, "PyFuncStateless": rnd_name}):
        return tf.py_func(func, inp, Tout, stateful=stateful, name=name)

# define simple model object that provides logits and gradients
class Model(object):

    def __init__(self):
        pass

    def logits(self, x):
        return x**2

    def gradient(self, x):
        return 2 * x

model = Model()

# wrap model predictions as TF op using py_func
def get_logits_op(x, name=None):
    print(x)
    with ops.name_scope(name, "logits", [x]) as name:
        sqr_x = py_func(model.logits,
                        [x],
                        [tf.float32],
                        name=name,
                        grad=_gradient_op)  # <-- here's the call to the gradient
        return sqr_x[0]

# wrap model gradient as TF op
def _gradient_op(op, grad):
    x = op.inputs[0]
    return model.gradient(x)

with tf.Session() as sess:
    x = tf.constant([1., 2.])
    logits = get_logits_op(x)
    tf.global_variables_initializer().run()

    print(x.eval(), logits.eval(), tf.gradients(logits, x)[0].eval())
