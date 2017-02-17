import nengo
import numpy as np
import pytest
import tensorflow as tf

from nengo_deeplearning.tests import TestSimulator


@pytest.mark.parametrize("device", ("/cpu:0", "/gpu:0"))
def test_gradients(device):
    step_blocks = 10
    minibatch_size = 20

    with nengo.Network() as net:
        inp = nengo.Node([0])
        inp2 = nengo.Node([0, 0])

        # sigmoid neurons
        ens = nengo.Ensemble(10, 1, neuron_type=nengo.Sigmoid())

        # normal decoded connection
        nengo.Connection(inp, ens)

        # # recurrent connection
        nengo.Connection(ens, ens)

        # rectified neurons
        ens2 = nengo.Ensemble(10, 2, neuron_type=nengo.RectifiedLinear())

        # neuron--neuron connection
        nengo.Connection(ens, ens2, transform=[[1], [1]],
                         solver=nengo.solvers.LstsqL2(weights=True))
        # nengo.Connection(ens.neurons, ens2.neurons)

        # sliced output, no synapse, transform
        nengo.Connection(inp, ens2[0], synapse=None, transform=0.5)

        # sliced input, sliced output
        nengo.Connection(inp2[0], ens2[1])

        p = nengo.Probe(ens)
        p2 = nengo.Probe(ens2)

    with TestSimulator(net, step_blocks=step_blocks, device=device,
                       minibatch_size=minibatch_size) as sim:
        sim.tensor_graph.build_optimizer(
            tf.train.GradientDescentOptimizer(0.1),
            {x: np.zeros((minibatch_size, step_blocks, x.size_in))
             for x in (p, p2)}, "mse")

        # fill in placeholder inputs
        feed_dict = {
            sim.tensor_graph.step_var: 0,
            sim.tensor_graph.stop_var: sim.step_blocks,
        }
        feed_dict.update(
            {k: v for k, v in zip(
                sim.tensor_graph.base_vars,
                [x[0] for x in sim.tensor_graph.base_arrays_init.values()])
             if k.op.type == "Placeholder"})
        feed_dict.update(
            {sim.tensor_graph.target_phs[x]:
                 np.zeros(sim.tensor_graph.target_phs[x].get_shape())
             for x in (p, p2)})

        # check gradient wrt inp
        x = sim.tensor_graph.invariant_ph[inp]
        feed_dict[sim.tensor_graph.invariant_ph[inp2]] = np.zeros(
            (step_blocks, inp2.size_out, minibatch_size))
        assert tf.test.compute_gradient_error(
            x, x.get_shape().as_list(), sim.tensor_graph.loss, (1,),
            extra_feed_dict=feed_dict) < 1e-4

        # check gradient wrt inp2
        x = sim.tensor_graph.invariant_ph[inp2]
        del feed_dict[sim.tensor_graph.invariant_ph[inp2]]
        feed_dict[sim.tensor_graph.invariant_ph[inp]] = np.zeros(
            (step_blocks, inp.size_out, minibatch_size))
        assert tf.test.compute_gradient_error(
            x, x.get_shape().as_list(), sim.tensor_graph.loss, (1,),
            extra_feed_dict=feed_dict) < 1e-4

        # also check the gradients this way, to make sure that check_gradients
        # is working
        sim.check_gradients()
