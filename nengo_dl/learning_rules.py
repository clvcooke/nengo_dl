from nengo.builder.learning_rules import SimBCM, SimOja, SimVoja
import numpy as np
import tensorflow as tf

from nengo_dl.builder import Builder, OpBuilder


@Builder.register(SimBCM)
class SimBCMBuilder(OpBuilder):
    """Build a group of :class:`~nengo:nengo.builder.learning_rules.SimBCM`
    operators."""

    def __init__(self, ops, signals):
        self.post_data = signals.combine([op.post_filtered for op in ops])
        self.theta_data = signals.combine([op.theta for op in ops])

        self.pre_data = signals.combine(
            [op.pre_filtered for op in ops
             for _ in range(op.post_filtered.shape[0])], load_indices=False)
        self.pre_data = self.pre_data.reshape((self.post_data.shape[0],
                                               ops[0].pre_filtered.shape[0]))
        self.pre_data.load_indices()

        self.learning_rate = tf.constant(
            [[op.learning_rate] for op in ops
             for _ in range(op.post_filtered.shape[0])],
            dtype=signals.dtype)

        self.output_data = signals.combine([op.delta for op in ops])

    def build_step(self, signals):
        pre = signals.gather(self.pre_data)
        post = signals.gather(self.post_data)
        theta = signals.gather(self.theta_data)

        post = self.learning_rate * signals.dt * post * (post - theta)

        post = tf.expand_dims(post, 1)

        signals.scatter(self.output_data, post * pre)


@Builder.register(SimOja)
class SimOjaBuilder(OpBuilder):
    """Build a group of :class:`~nengo:nengo.builder.learning_rules.SimOja`
        operators."""

    def __init__(self, ops, signals):
        self.post_data = signals.combine([op.post_filtered for op in ops])

        self.pre_data = signals.combine(
            [op.pre_filtered for op in ops
             for _ in range(op.post_filtered.shape[0])], load_indices=False)
        self.pre_data = self.pre_data.reshape((self.post_data.shape[0],
                                               ops[0].pre_filtered.shape[0]))
        self.pre_data.load_indices()

        self.weights_data = signals.combine([op.weights for op in ops])
        self.output_data = signals.combine([op.delta for op in ops])

        self.learning_rate = tf.constant(
            [[[op.learning_rate]] for op in ops
             for _ in range(op.post_filtered.shape[0])],
            dtype=signals.dtype)

        self.beta = tf.constant(
            [[[op.beta]] for op in ops for _ in
             range(op.post_filtered.shape[0])],
            dtype=signals.dtype)

    def build_step(self, signals):
        pre = signals.gather(self.pre_data)
        post = signals.gather(self.post_data)
        weights = signals.gather(self.weights_data)

        post = tf.expand_dims(post, 1)

        alpha = self.learning_rate * signals.dt

        update = alpha * post ** 2
        update *= -self.beta * weights
        update += alpha * post * pre

        signals.scatter(self.output_data, update)


@Builder.register(SimVoja)
class SimVojaBuilder(OpBuilder):
    """Build a group of :class:`~nengo:nengo.builder.learning_rules.SimVoja`
        operators."""

    def __init__(self, ops, signals):
        self.post_data = signals.combine([op.post_filtered for op in ops])

        self.pre_data = signals.combine(
            [op.pre_decoded for op in ops
             for _ in range(op.post_filtered.shape[0])], load_indices=False)
        self.pre_data = self.pre_data.reshape((self.post_data.shape[0],
                                               ops[0].pre_decoded.shape[0]))
        self.pre_data.load_indices()

        self.learning_data = signals.combine(
            [op.learning_signal for op in ops
             for _ in range(op.post_filtered.shape[0])])
        self.encoder_data = signals.combine([op.scaled_encoders for op in ops])
        self.output_data = signals.combine([op.delta for op in ops])

        self.scale = tf.constant(
            np.concatenate([op.scale[:, None, None] for op in ops], axis=0),
            dtype=signals.dtype)

        self.learning_rate = tf.constant(
            [[op.learning_rate] for op in ops
             for _ in range(op.post_filtered.shape[0])], dtype=signals.dtype)

    def build_step(self, signals):
        pre = signals.gather(self.pre_data)
        post = signals.gather(self.post_data)
        learning_signal = signals.gather(self.learning_data)
        scaled_encoders = signals.gather(self.encoder_data)

        alpha = tf.expand_dims(
            self.learning_rate * signals.dt * learning_signal, 1)
        post = tf.expand_dims(post, 1)

        update = alpha * (self.scale * post * pre - post * scaled_encoders)

        signals.scatter(self.output_data, update)
