import nengo
from nengo.exceptions import ValidationError
import numpy as np
import pytest
import tensorflow as tf

import nengo_deeplearning as nengo_dl


def test_validation():
    with nengo.Network():
        with pytest.raises(ValidationError):
            nengo_dl.TensorNode([0])

        with pytest.raises(ValidationError):
            nengo_dl.TensorNode(lambda t: t, size_out=0)

        with pytest.raises(ValidationError):
            nengo_dl.TensorNode(lambda a, b, c: a)

        with pytest.raises(ValidationError):
            nengo_dl.TensorNode(lambda x: None)

        with pytest.raises(ValidationError):
            nengo_dl.TensorNode(lambda x: [0])

        with pytest.raises(ValidationError):
            nengo_dl.TensorNode(lambda t: tf.zeros((2, 2, 2)))

        n = nengo_dl.TensorNode(lambda t: tf.zeros((5, 2)))
        assert n.size_out == 2

        n = nengo_dl.TensorNode(lambda t: tf.zeros((5, 2)), size_out=4)
        assert n.size_out == 4


def test_node():
    minibatch_size=3
    with nengo.Network() as net:
        node0 = nengo_dl.TensorNode(lambda t: tf.tile(tf.reshape(t, (1, -1)),
                                            (minibatch_size, 1)))
        node1 = nengo_dl.TensorNode(lambda t, x: tf.sin(x), size_in=1)
        nengo.Connection(node0, node1, synapse=None)

        p0 = nengo.Probe(node0)
        p1 = nengo.Probe(node1)

    with nengo_dl.Simulator(net, minibatch_size=minibatch_size) as sim:
        sim.run_steps(10)

    assert np.allclose(sim.data[p0], sim.trange()[None, :, None])
    assert np.allclose(sim.data[p1], np.sin(sim.trange()[None, :, None]))
