import warnings

from nengo.builder.signal import Signal
from nengo.exceptions import BuildError
from nengo.neurons import Direct
import numpy as np
import tensorflow as tf

from nengo_deeplearning import DEBUG


class TensorSignal(object):
    """Represents a tensor as an indexed view into a base variable.

    Parameters
    ----------
    indices : tuple or list or :class:`~numpy:numpy.ndarray` of int
        indices along the first axis of the base array corresponding to the
        data for this signal
    key : tuple
        dtype and shape of base array, used as key to find the base
    display_shape : tuple of int, optional
        view shape of this signal (may differ from shape of base array)
    label : str, optional
        name for this signal, used to make debugging easier
    """

    def __init__(self, indices, key, display_shape=None, label="TensorSignal"):
        # make indices read-only
        assert isinstance(indices, (tuple, list, np.ndarray))
        self._indices = np.asarray(indices)
        self._indices.flags.writeable = False
        self.tf_indices = None

        self.key = key

        self.display_shape = display_shape

        self.label = label

    @property
    def indices(self):
        return self._indices

    @indices.setter
    def indices(self, val):
        raise BuildError("Indices are read only")

    @property
    def dtype(self):
        return self.key[0]

    @property
    def base_shape(self):
        return self.key[1]

    @property
    def trainable(self):
        return self.key[2]

    @property
    def minibatched(self):
        return not self.trainable

    @property
    def shape(self):
        if self.display_shape is None:
            return (len(self.indices),) + self.base_shape[1:]
        else:
            return self.display_shape

    @property
    def ndim(self):
        return len(self.shape)

    def __repr__(self):
        return "TensorSignal(key=%s, shape=%s, label=%s)" % (
            self.key, self.shape, self.label)

    def __getitem__(self, indices):
        """Create a new TensorSignal representing a subset (slice or advanced
        indexing) of the indices of this TensorSignal."""

        if indices is Ellipsis:
            return self

        return TensorSignal(self.indices[indices], self.key,
                            label=self.label + ".slice")

    def reshape(self, shape):
        """Create a new TensorSignal representing a reshaped view of the
        same data as this TensorSignal."""

        # replace -1 with inferred dimension
        assert shape.count(-1) <= 1
        n_elem = np.prod(self.shape)
        n_shape = np.prod([x for x in shape if x != -1])
        assert n_elem % n_shape == 0
        shape = tuple([x if x != -1 else n_elem // n_shape for x in shape])

        if np.prod(shape) != np.prod(self.shape):
            raise BuildError("Number of elements don't match in reshape")

        return TensorSignal(
            self.indices, self.key, display_shape=shape,
            label=self.label + ".reshape(%s)" % (shape,))

    def load_indices(self):
        """Loads the indices for this signal into tensorflow, and if the
        indices form a contiguous slice then also loads the start/stop/step of
        that slice."""

        self.tf_indices = tf.constant(self.indices)

        start = self.indices[0]
        stop = self.indices[-1] + 1
        step = (self.indices[1] - self.indices[0] if len(self.indices) > 1
                else 1)
        if step != 0 and np.all(self.indices == np.arange(start, stop, step)):
            self.as_slice = (tf.constant([start]), tf.constant([stop]),
                             tf.constant([step]))
        else:
            self.as_slice = None

    def broadcast(self, axis, length):
        assert axis in (0, -1)

        indices = self.indices
        indices = np.stack([indices] * length, axis=axis)
        indices = np.reshape(indices, (-1,))

        if axis == -1:
            display_shape = self.shape + (length,)
        else:
            display_shape = (length,) + self.shape

        return TensorSignal(
            indices, self.key, display_shape=display_shape,
            label=self.label + ".broadcast(%d, %d)" % (axis, length))

    # def tile(self, length):
    #     assert self.in_tf
    #
    #     # repeat along the first axis the given number of times
    #     # note: we don't use tf.tile because it doesn't have a GPU kernel
    #     indices = tf.concat(0, [self.indices] * length)
    #     shape = (self.shape[0] * length,) + self.shape[1:]
    #
    #     return TensorSignal(
    #         indices, self.key, display_shape=shape,
    #         label=self.label + ".tile(%d)" % length)


class SignalDict(object):
    """Map from Signal -> Tensor

    Takes care of scatter/gather logic to read/write signals within the base
    arrays.

    Parameters
    ----------
    sig_map : dict of {`nengo.builder.signal.Signal`: `TensorSignal`}
        mapping from `nengo` signals to `nengo_dl` signals
    dtype : tf.DType
        floating point precision used in signals
    dt : float
        simulation timestep
    minibatch_size : int
        number of items in each minibatch
    """

    def __init__(self, sig_map, dtype, dt, minibatch_size):
        self.dtype = dtype
        self.sig_map = sig_map
        self.minibatch_size = minibatch_size

        # create this constant once here so we don't end up creating a new
        # dt constant in each operator
        self.dt = tf.constant(dt, dtype)
        self.dt.dt_val = dt  # store the actual value as well

    def scatter(self, dst, val, mode="update"):
        """Updates the base data corresponding to `dst`.

        Parameters
        ----------
        dst : :class:`.TensorSignal`
            signal indicating the data to be modified in base array
        val : `tf.Tensor`
            update data (same shape as `dst`, i.e. a dense array <= the size of
            the base array)
        mode: "update" or "inc" or "mul"
            overwrite/add/multiply the data at `dst` with `val`
        """

        assert dst.tf_indices is not None

        if val.dtype.is_floating and val.dtype.base_dtype != self.dtype:
            raise BuildError("Tensor detected with wrong dtype (%s), should "
                             "be %s." % (val.dtype.base_dtype, self.dtype))

        # align val shape with dst base shape
        dst_shape = (dst.shape[0],) + dst.base_shape[1:]
        if dst.minibatched:
            dst_shape += (self.minibatch_size,)
        if val.get_shape().ndims != len(dst_shape):
            val = tf.reshape(val, dst_shape)

        if mode == "update":
            scatter_f = tf.scatter_update
        elif mode == "inc":
            scatter_f = tf.scatter_add
        elif mode == "mul":
            scatter_f = tf.scatter_mul

        if DEBUG:
            print("scatter")
            print("values", val)
            print("dst", dst)
            print("indices", dst.indices)
            print("dst base", self.bases[dst.key])
            print("reads_by_base", self.reads_by_base[self.bases[dst.key]])

        # make sure that any reads to the target signal happen before this
        # write (note: this is only any reads that have happened since the
        # last write, since each write changes the base array object)
        with tf.control_dependencies(self.reads_by_base[self.bases[dst.key]]):
            self.bases[dst.key] = scatter_f(
                self.bases[dst.key], dst.tf_indices, val)

        if DEBUG:
            print("new dst base", self.bases[dst.key])

    def gather(self, src, force_copy=False):
        """Fetches the data corresponding to `src` from the base array.

        Parameters
        ----------
        src : :class:`.TensorSignal`
            signal indicating the data to be read from base array
        force_copy : bool, optional
            if True, always perform a gather, not a slice (this forces a
            copy). note that setting force_copy=False does not guarantee that
            a copy won't be performed.

        Returns
        -------
        `tf.Tensor`
            tensor object corresponding to a dense subset of data from the
            base array
        """

        assert src.tf_indices is not None

        if DEBUG:
            print("gather")
            print("src", src)
            print("indices", src.indices)
            print("src base", self.bases[src.key])

        # we prefer to get the data via `strided_slice` if possible, as it
        # is more efficient
        if force_copy or src.as_slice is None:
            result = tf.gather(self.bases[src.key], src.tf_indices)
        else:
            result = tf.strided_slice(self.bases[src.key], *src.as_slice)

        # reshape the data according to the shape set in `src`, if there is
        # one, otherwise keep the shape of the base array
        src_shape = src.shape
        if src.minibatched:
            src_shape += (self.minibatch_size,)
        if src.display_shape is not None:
            result = tf.reshape(result, src_shape)
        else:
            result.set_shape(src_shape)

        # whenever we read from an array we use this to mark it as "read"
        # (so that any future writes to the array will be scheduled after
        # the read)
        # TODO: we could store the indices as well, so that future writes are
        # only delayed if they write to the same part of the array
        self.reads_by_base[self.bases[src.key]] += [result]

        return result

    def combine(self, sigs, load_indices=True):
        """Concatenates several TensorSignals into one by concatenating along
        the first axis.

        Parameters
        ----------
        list of :class:`.TensorSignal` or `nengo.builder.signal.Signal`
            signals to be combined
        load_indices : bool, optional
            if True, load the indices for the new signal into tensorflow right
            away (otherwise they will need to be manually loaded later)

        Returns
        -------
        :class:`.TensorSignal`
            new `TensorSignal` representing the concatenation of the data in
            `sigs`
        """

        if len(sigs) == 0:
            return []

        assert isinstance(sigs, (list, tuple))
        assert isinstance(sigs[0], (Signal, TensorSignal))

        sigs = [self.sig_map[s] if isinstance(s, Signal) else s for s in sigs]

        key = sigs[0].key
        # make sure all the signals have the same base
        assert all([s.key == key for s in sigs])

        indices = np.concatenate([s.indices for s in sigs], axis=0)

        # check if any of the signals have been reshaped
        if np.any([s.display_shape is not None for s in sigs]):
            # make sure they all have the same shape for axes > 0 (they're
            # concatenated along the first dimension)
            assert all([s.shape[1:] == sigs[0].shape[1:] for s in sigs])

            shape = (np.sum([s.shape[0] for s in sigs]),) + sigs[0].shape[1:]
        else:
            shape = None

        output = TensorSignal(indices, key, display_shape=shape)

        if load_indices:
            output.load_indices()

        return output

    def __str__(self):
        """Pretty-print the signals and current values."""

        return "\n".join(["%s: %s" % (repr(k), repr(self[k]))
                          for k in self])


def mark_signals(model):
    # TODO: documentation/tests

    if model.toplevel is None:
        warnings.warn("No top-level network in model")
    else:
        for ens in model.toplevel.all_ensembles:
            model.sig[ens]["encoders"].trainable = True

            if not isinstance(ens.neuron_type, Direct):
                model.sig[ens.neurons]["bias"].trainable = True

        for conn in model.toplevel.all_connections:
            # note: this doesn't include probe connections, since they aren't
            # added to the network
            # TODO: should we disable training on connections to learning
            # rules?
            model.sig[conn]["weights"].trainable = True

    # mark everything as not trainable by default
    for op in model.operators:
        for sig in op.all_signals:
            if not hasattr(sig, "trainable"):
                sig.trainable = False
            sig.minibatched = not sig.trainable
