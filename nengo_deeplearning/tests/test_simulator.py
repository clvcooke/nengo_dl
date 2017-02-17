import nengo
import numpy as np
import pytest
import tensorflow as tf

from nengo_deeplearning.tests import TestSimulator


def test_persistent_state():
    """Make sure that state is preserved between runs."""

    with nengo.Network(seed=0) as net:
        inp = nengo.Node([1])
        ens = nengo.Ensemble(1000, 1)
        nengo.Connection(inp, ens)
        p = nengo.Probe(ens)

    with TestSimulator(net, step_blocks=5) as sim:
        sim.run_steps(100)
        data = sim.data[p]
        sim.reset()

        sim.run_steps(100)
        data2 = sim.data[p]
        sim.reset()

        for _ in range(20):
            sim.run_steps(5)
        data3 = sim.data[p]

    assert np.allclose(data, data2)
    assert np.allclose(data2, data3)


def test_step_blocks():
    with nengo.Network(seed=0) as net:
        inp = nengo.Node(np.sin)
        ens = nengo.Ensemble(10, 1)
        nengo.Connection(inp, ens)
        p = nengo.Probe(ens)

    with TestSimulator(net, step_blocks=25) as sim1:
        sim1.run_steps(50)
    with TestSimulator(net, step_blocks=10) as sim2:
        sim2.run_steps(50)
    with TestSimulator(net, unroll_simulation=False, step_blocks=None) as sim3:
        sim3.run_steps(50)

    assert np.allclose(sim1.data[p], sim2.data[p])
    assert np.allclose(sim2.data[p], sim3.data[p])


def test_unroll_simulation():
    # note: we run this multiple times because the effects of unrolling can
    # be somewhat stochastic depending on the op order
    for _ in range(10):
        with nengo.Network(seed=0) as net:
            inp = nengo.Node(np.sin)
            ens = nengo.Ensemble(10, 1)
            nengo.Connection(inp, ens)
            p = nengo.Probe(ens)

        with TestSimulator(net, step_blocks=10,
                           unroll_simulation=False) as sim1:
            sim1.run_steps(50)

        with TestSimulator(net, step_blocks=10,
                           unroll_simulation=True) as sim2:
            sim2.run_steps(50)

        assert np.allclose(sim1.data[p], sim2.data[p])


def test_minibatch():
    with nengo.Network(seed=0) as net:
        inp = [nengo.Node(output=[0.5]), nengo.Node(output=np.sin),
               nengo.Node(output=nengo.processes.WhiteSignal(5, 0.5, seed=0))]

        ens = [
            nengo.Ensemble(10, 1, neuron_type=nengo.AdaptiveLIF()),
            nengo.Ensemble(10, 1, neuron_type=nengo.LIFRate()),
            nengo.Ensemble(10, 2, noise=nengo.processes.WhiteNoise(seed=0))]

        nengo.Connection(inp[0], ens[0])
        nengo.Connection(inp[1], ens[1], synapse=None)
        nengo.Connection(inp[2], ens[2], synapse=nengo.Alpha(0.1),
                         transform=[[1], [1]])
        conn = nengo.Connection(ens[0], ens[1], learning_rule_type=nengo.PES())
        nengo.Connection(inp[0], conn.learning_rule)

        ps = [nengo.Probe(e) for e in ens]

    with TestSimulator(net, minibatch_size=None) as sim:
        probe_data = [[] for _ in ps]
        for i in range(5):
            sim.run_steps(1000)

            for j, p in enumerate(ps):
                probe_data[j] += [sim.data[p]]

            sim.reset()

        probe_data = [np.stack(x, axis=0) for x in probe_data]

    with TestSimulator(net, minibatch_size=5) as sim:
        sim.run_steps(1000)

    assert np.allclose(sim.data[ps[0]], probe_data[0])
    assert np.allclose(sim.data[ps[1]], probe_data[1])
    assert np.allclose(sim.data[ps[2]], probe_data[2])


def test_input_feeds():
    minibatch_size = 10
    step_blocks = 5

    with nengo.Network() as net:
        inp = nengo.Node([0, 0, 0])
        out = nengo.Node(size_in=3)
        nengo.Connection(inp, out, synapse=None)
        p = nengo.Probe(out)

    with TestSimulator(net, minibatch_size=minibatch_size,
                       step_blocks=step_blocks) as sim:
        val = np.random.randn(minibatch_size, step_blocks, 3)
        sim.run_steps(step_blocks, input_feeds={inp: val})
        assert np.allclose(sim.data[p], val)

        with pytest.raises(nengo.exceptions.SimulationError):
            sim.run_steps(step_blocks, input_feeds={
                inp: np.random.randn(minibatch_size, step_blocks + 1, 3)})


def test_train_ff():
    minibatch_size = 4
    step_blocks = 1

    with nengo.Network() as net:
        net.config[nengo.Ensemble].gain = nengo.dists.Choice([1])
        net.config[nengo.Ensemble].bias = nengo.dists.Uniform(-1, 1)
        net.config[nengo.Connection].synapse = None

        inp = nengo.Node([0, 0])
        ens = nengo.Ensemble(10, 1, neuron_type=nengo.Sigmoid())
        out = nengo.Ensemble(1, 1, neuron_type=nengo.Sigmoid())
        nengo.Connection(inp, ens.neurons,
                         transform=np.random.uniform(-0.1, 0.1, size=(10, 2)))
        nengo.Connection(ens, out, solver=nengo.solvers.LstsqL2(weights=True))
        p = nengo.Probe(out)

    with TestSimulator(net, minibatch_size=minibatch_size,
                       step_blocks=step_blocks) as sim:
        x = np.asarray([[[0, 0]], [[0, 1]], [[1, 0]], [[1, 1]]])
        y = np.asarray([[[0]], [[1]], [[1]], [[0]]])

        sim.print_params("pre")

        sim.train({inp: x}, {p: y}, tf.train.GradientDescentOptimizer(1e-3),
                  n_epochs=10000)

        sim.print_params("post")

        sim.check_gradients()

        sim.step(input_feeds={inp: x})

        assert np.allclose(sim.data[p], y)


def test_train_recurrent():
    pass


def test_train_objective():
    pass


def test_generate_inputs():
    pass


def test_update_probe_data():
    pass


def test_save_load_params():
    pass
