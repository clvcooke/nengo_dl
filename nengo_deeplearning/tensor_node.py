from nengo import Node, builder
from nengo.base import NengoObject
from nengo.builder.operator import Reset
from nengo.exceptions import ValidationError, SimulationError
from nengo.params import Default, IntParam, Parameter
import numpy as np
import tensorflow as tf

from nengo_deeplearning.builder import Builder, OpBuilder


# TODO: documentation

class TensorFuncParam(Parameter):
    def __init__(self, name, readonly=False):
        super(TensorFuncParam, self).__init__(
            name, optional=False, readonly=readonly)

    def __set__(self, node, output):
        super(TensorFuncParam, self).validate(node, output)

        node.size_in = 0 if node.size_in is None else node.size_in

        # We trust user's size_out if set, because calling output
        # may have unintended consequences
        if node.size_out is None:
            node.size_out = self.check_size_out(node, output)

        # --- Set output
        self.data[node] = output

    def check_size_out(self, node, output):
        if not callable(output):
            raise ValidationError("TensorNode output must be a function",
                                  attr=self.name, obj=node)

        t, x = tf.constant(0.0), tf.zeros((1, node.size_in))
        args = (t, x) if node.size_in > 0 else (t,)
        try:
            result = output(*args)
        except Exception as e:
            msg = ("output function '%s' is expected to accept exactly "
                   "%d argument" % (output, len(args)))
            msg += (' (time, as a float)' if len(args) == 1 else
                    's (time, as a float and data, as a tf.Tensor)')
            raise ValidationError(msg, attr=self.name, obj=node)

        if not isinstance(result, tf.Tensor):
            raise ValidationError("TensorNode function must return a Tensor",
                                  attr=self.name, obj=node)

        if result.get_shape().ndims != 2:
            raise ValidationError("Node output must be a minibatched vector "
                                  "(got shape %s)" % result.get_shape(),
                                  attr=self.name, obj=node)

        return result.get_shape()[1].value


class TensorNode(Node):
    tensor_func = TensorFuncParam('tensor_func')
    size_in = IntParam('size_in', default=None, low=0, optional=True)
    size_out = IntParam('size_out', default=None, low=1, optional=True)

    def __init__(self, tensor_func, size_in=Default, size_out=Default,
                 label=Default, seed=Default):
        # note: we bypass the Node constructor, because we don't want to
        # perform validation on `output`
        NengoObject.__init__(self, label=label, seed=seed)

        self.size_in = size_in
        self.size_out = size_out
        self.tensor_func = tensor_func


@builder.Builder.register(TensorNode)
def build_tensor_node(model, node):
    # input signal
    if node.size_in > 0:
        sig_in = builder.Signal(np.zeros(node.size_in), name="%s.in" % node)
        model.add_op(Reset(sig_in))
    else:
        sig_in = None

    sig_out = builder.Signal(np.zeros(node.size_out), name="%s.out" % node)

    model.sig[node]['in'] = sig_in
    model.sig[node]['out'] = sig_out
    model.params[node] = None

    model.add_op(SimTensorNode(node.tensor_func, sig_in, sig_out))


class SimTensorNode(builder.Operator):
    def __init__(self, func, input, output, tag=None):
        super(SimTensorNode, self).__init__(tag=tag)

        self.func = func
        self.input = input
        self.output = output

        self.sets = [output]
        self.incs = []
        self.reads = [] if input is None else [input]
        self.updates = []

    def make_step(self, *args, **kwargs):
        def error():
            raise SimulationError("TensorNode can only be simulated in the "
                                  "NengoDL simulator")

        return error


@Builder.register(SimTensorNode)
class SimTensorNodeBuilder(OpBuilder):
    def __init__(self, ops, signals):
        assert len(ops) == 1
        op = ops[0]

        self.func = op.func

        if op.input is None:
            self.src_data = None
        else:
            self.src_data = signals.sig_map[op.input]
            self.src_data.load_indices()
            assert self.src_data.ndim == 1

        self.dst_data = signals.sig_map[op.output]
        self.dst_data.load_indices()

    def build_step(self, signals):
        if self.src_data is None:
            output = self.func(signals.time)
        else:
            input = signals.gather(self.src_data)

            # move minibatch dimension to front
            input = tf.transpose(input, (1, 0))

            output = self.func(signals.time, input)

        # move minibatch dimension back to end
        output_dim = output.get_shape().ndims - 1
        output = tf.transpose(
            output, [output_dim] + [i for i in range(output_dim)])

        signals.scatter(self.dst_data, output)
