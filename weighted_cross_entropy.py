import mxnet as mx
from mxnet.gluon.loss import Loss

class Weighted_Categorical_CrossEntropy(Loss):
        """
        A weighted version of keras.objectives.categorical_crossentropy

        Variables:
            weights: numpy array of shape (C,) where C is the number of classes

        Usage:
            weights = np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
            loss = weighted_categorical_crossentropy(weights)
            model.compile(loss=loss,optimizer='adam')
        """
        def __init__(self, weights, axis=1, _batch_axis=0, **kwards):
            Loss.__init__(self, **kwards)

            self.weights = mx.nd.array(weights)

        def hybrid_forward(self, F, y_true, y_pred):
            # scale predictions so that the class probas of each sample sum to 1
            y_pred /= F.sum(y_pred, axis=-1)
            # clip to prevent NaN's and Inf's
            y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
            loss = y_true * K.log(y_pred) * weights
            loss = -K.sum(loss, -1)
            # loss = K.mean(loss, axis=[1,2])
            # print(loss.shape)
            return loss
        return loss
