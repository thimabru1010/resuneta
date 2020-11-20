import mxnet as mx
from mxnet.gluon.loss import Loss, is_np_array


def _reshape_like(F, x, y):
    """Reshapes x to the same shape as y."""
    if F is mx.ndarray:
        return x.reshape(y.shape)
    elif is_np_array():
        F = F.npx
    return F.reshape_like(x, y)


# def _apply_weighting(F, loss, weight=None, sample_weight=None):
#     """Apply weighting to loss.
#
#     Parameters
#     ----------
#     loss : Symbol
#         The loss to be weighted.
#     weight : float or None
#         Global scalar weight for loss.
#     sample_weight : Symbol or None
#         Per sample weighting. Must be broadcastable to
#         the same shape as loss. For example, if loss has
#         shape (64, 10) and you want to weight each sample
#         in the batch separately, `sample_weight` should have
#         shape (64, 1).
#
#     Returns
#     -------
#     loss : Symbol
#         Weighted loss
#     """
#     if sample_weight is not None:
#         if is_np_array():
#             loss = loss * sample_weight
#         else:
#             loss = F.broadcast_mul(loss, sample_weight)
#
#     if weight is not None:
#         assert isinstance(weight, numeric_types), "weight must be a number"
#         loss = loss * weight
#
#     return loss


class WeightedSoftmaxCrossEntropyLoss(Loss):
    r"""Computes the softmax cross entropy loss. (alias: SoftmaxCELoss)

    If `sparse_label` is `True` (default), label should contain integer
    category indicators:

    .. math::

        \DeclareMathOperator{softmax}{softmax}

        p = \softmax({pred})

        L = -\sum_i \log p_{i,{label}_i}

    `label`'s shape should be `pred`'s shape with the `axis` dimension removed.
    i.e. for `pred` with shape (1,2,3,4) and `axis = 2`, `label`'s shape should
    be (1,2,4).

    If `sparse_label` is `False`, `label` should contain probability distribution
    and `label`'s shape should be the same with `pred`:

    .. math::

        p = \softmax({pred})

        L = -\sum_i \sum_j {label}_j \log p_{ij}

    Parameters
    ----------
    axis : int, default -1
        The axis to sum over when computing softmax and entropy.
    sparse_label : bool, default True
        Whether label is an integer array instead of probability distribution.
    from_logits : bool, default False
        Whether input is a log probability (usually from log_softmax) instead
        of unnormalized numbers.
    weight : float or None
        Global scalar weight for loss.
    batch_axis : int, default 0
        The axis that represents mini-batch.
    class_weights : mx.nd.array, default None
        Weights to apply on classes.


    Inputs:
        - **pred**: the prediction tensor, where the `batch_axis` dimension
          ranges over batch size and `axis` dimension ranges over the number
          of classes.
        - **label**: the truth tensor. When `sparse_label` is True, `label`'s
          shape should be `pred`'s shape with the `axis` dimension removed.
          i.e. for `pred` with shape (1,2,3,4) and `axis = 2`, `label`'s shape
          should be (1,2,4) and values should be integers between 0 and 2. If
          `sparse_label` is False, `label`'s shape must be the same as `pred`
          and values should be floats in the range `[0, 1]`.
        - **sample_weight**: element-wise weighting tensor. Must be broadcastable
          to the same shape as label. For example, if label has shape (64, 10)
          and you want to weigh each sample in the batch separately,
          sample_weight should have shape (64, 1).

    Outputs:
        - **loss**: loss tensor with shape (batch_size,). Dimenions other than
          batch_axis are averaged out.
    """

    def __init__(self, axis=-1, sparse_label=True, from_logits=False, weight=None,
                 batch_axis=0, class_weights=None, **kwargs):
        super(WeightedSoftmaxCrossEntropyLoss, self).__init__(
            weight, batch_axis, **kwargs)
        self._axis = axis
        self._sparse_label = sparse_label
        self._from_logits = from_logits
        print(f'Checing WCE from logits: {self._from_logits}')
        # self._class_weights = mx.nd.array(class_weights)
        self._class_weights = class_weights

    def hybrid_forward(self, F, pred, label, sample_weight=None):
        if is_np_array():
            log_softmax = F.npx.log_softmax
            pick = F.npx.pick
        else:
            log_softmax = F.log_softmax
            pick = F.pick
        if not self._from_logits:
            # pred = log_softmax(pred, self._axis)
            pred = F.log_softmax(pred, self._axis)
            print(pred)
        if self._sparse_label:
            loss = -pick(pred, label, axis=self._axis, keepdims=True)
        else:
            label = _reshape_like(F, label, pred)
            res = pred * label
            # Convert tensor from shape B x C x H x W --> B  x H x W x C
            wres = res.transpose((0, 2, 3, 1)) * self._class_weights.copyto(res.ctx)
            # get back to original shape
            wres = wres.transpose((0, 3, 1, 2))
            loss = -(wres).sum(axis=self._axis, keepdims=True)
        # loss = _apply_weighting(F, loss, self._weight, sample_weight)
        # if is_np_array():
        if F is mx.ndarray:
            return loss.mean(axis=tuple(range(1, loss.ndim)))
        else:
            return F.npx.batch_flatten(loss).mean(axis=1)
        # else:
        #     return loss.mean(axis=self._batch_axis, exclude=True)
