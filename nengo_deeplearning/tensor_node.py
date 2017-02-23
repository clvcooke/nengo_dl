from nengo import Node, builder
from nengo.base import NengoObject
from nengo.builder.operator import Reset
from nengo.exceptions import ValidationError, SimulationError
from nengo.params import Default, IntParam
import numpy as np
import tensorflow as tf

from nengo_deeplearning.builder import Builder, OpBuilder


class TensorNode(Node):
    size_in = IntParam('size_in', default=None, low=0, optional=True)

    def __init__(self, tensor_func, size_out, size_in=Default, label=Default,
                 seed=Default):
        # note: we bypass the Node constructor, because we don't want to
        # perform validation on `output`
        NengoObject.__init__(self, label=label, seed=seed)

        self.tensor_func = tensor_func
        self.size_in = size_in
        self.size_out = size_out

        if not callable(tensor_func):
            raise ValidationError("TensorNode output must be a function",
                                  "output")

        if size_out <= 0:
            raise ValidationError("TensorNode size_out must be > 0",
                                  "size_out")

        if self.size_in > 0:
            x = tensor_func(tf.constant(0.), tf.zeros((self.size_in, 1)))
        else:
            x = tensor_func(tf.constant(0.))

        if not isinstance(x, tf.Tensor):
            raise ValidationError("TensorNode function must return a Tensor",
                                  "output")

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

        self.dst_data = signals.sig_map[op.output]
        self.dst_data.load_indices()

    def build_step(self, signals):
        if self.src_data is None:
            output = self.func(signals.time)
        else:
            input = signals.gather(self.src_data)
            output = self.func(signals.time, input)

        signals.scatter(self.dst_data, output)
