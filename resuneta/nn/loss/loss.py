import numpy as np
from mxnet.gluon.loss import Loss
import mxnet as mx


class Tanimoto(Loss):
    def __init__(self, no_past_def, _smooth=1.0e-5, _axis=[2,3], _weight=None, _batch_axis=0,
                 **kwards):
        Loss.__init__(self, weight=_weight, batch_axis=_batch_axis, **kwards)

        self.axis = _axis
        self.smooth = _smooth
        self.weight = _weight

        # self.class_weights = class_weights
        self.no_past_def = no_past_def

    def hybrid_forward(self, F, _preds, _label):

        # Evaluate the mean volume of class per batch
        Vli = F.mean(F.sum(_label, axis=self.axis), axis=0)
        # wli =  1.0/Vli**2 # weighting scheme
        wli = F.reciprocal(Vli**2)  # weighting scheme

        # ---------------------This line is taken from niftyNet package --------------
        # ref: https://github.com/NifTK/NiftyNet/blob/dev/niftynet/layer/loss_segmentation.py, lines:170 -- 172
        # new_weights = tf.where(tf.is_inf(weights), tf.zeros_like(weights), weights)
        # weights = tf.where(tf.is_inf(weights), tf.ones_like(weights) * tf.reduce_max(new_weights), weights)
        # --------------------------------------------------------------------

        # ***********************************************************************************************
        # First turn inf elements to zero, then replace that with the maximum weight value
        new_weights = F.where(wli == np.float('inf'), F.zeros_like(wli), wli)
        wli = F.where(wli == np.float('inf'), F.broadcast_mul(F.ones_like(wli), F.max(new_weights)), wli)
        # ************************************************************************************************

        # print(wli.shape)
        # print(f'Actual: {wli}')
        # if self.no_past_def:
        #     no_consider = mx.nd.array([1.0, 1.0, 0.0], ctx=wli.ctx)
        #     wli = wli * no_consider
        #
        #     no_consider = mx.nd.array([1.0, 1.0, 0.0], ctx=_preds.ctx)
        #     _preds = no_consider * _preds.transpose((0, 2, 3, 1))
        #     _preds = _preds.transpose((0, 3, 1, 2))

        rl_x_pl = F.sum(F.broadcast_mul(_label, _preds), axis=self.axis)
        # print(f'rl_x_pl: {rl_x_pl.shape}')
        # This is sum of squares
        l = F.sum(F.broadcast_mul(_label, _label), axis=self.axis)
        r = F.sum(F.broadcast_mul(_preds, _preds), axis=self.axis)
        # print(f'{l.shape}')
        # print(f'{r.shape}')

        rl_p_pl = l + r - rl_x_pl

        if self.no_past_def:
            tnmt = (F.sum(F.broadcast_mul(wli, rl_x_pl)[:, :2], axis=1) + self.smooth) / (F.sum(F.broadcast_mul(wli, (rl_p_pl))[:, :2], axis=1) + self.smooth)

            return tnmt # This returns the tnmt for EACH data point, i.e. a vector of values equal to the batch size
        else:
            tnmt = (F.sum(F.broadcast_mul(wli, rl_x_pl), axis=1) + self.smooth) / (F.sum(F.broadcast_mul(wli, (rl_p_pl)), axis=1) + self.smooth)

            return tnmt # This returns the tnmt for EACH data point, i.e. a vector of values equal to the batch size

# This is the loss used in the manuscript of resuneta


class Tanimoto_with_dual(Loss):
    """
    Tanimoto coefficient with dual from: Diakogiannis et al 2019 (https://arxiv.org/abs/1904.00592)
    Note: to use it in deep learning training use: return 1. - 0.5*(loss1+loss2)
    """
    def __init__(self, _smooth=1.0e-5, _axis=[2, 3], _weight=None, _batch_axis=0
                 , no_past_def=True, **kwards):
        Loss.__init__(self, weight=_weight, batch_axis=_batch_axis, **kwards)

        with self.name_scope():
            self.Loss = Tanimoto(no_past_def=no_past_def,
                                 _smooth=_smooth, _axis=_axis)

    def hybrid_forward(self, F, _preds, _label):

        # measure of overlap
        loss1 = self.Loss(_preds, _label)

        # measure of non-overlap as inner product
        preds_dual = 1.0-_preds
        labels_dual = 1.0-_label
        loss2 = self.Loss(preds_dual, labels_dual)

        return 1 - 0.5*(loss1+loss2)
