import tensorflow as tf

class MedianSquaredError(tf.keras.losses.Loss):
    """ Class for computing the Median Squared Error (MedSE) for predictions."""
    def __init__(self):
        super(MedianSquaredError, self).__init__()

    def call(self, y_true, y_pred):
        """ Parameters
        ----------
        y_true:
            Ground truth values. shape = `[batch_size, d0, .. dN]`, except sparse loss functions such as sparse categorical crossentropy where shape = `[batch_size, d0, .. dN-1]`
        y_pred:
            The predicted values. shape = `[batch_size, d0, .. dN]`
        """
        se = tf.math.square(y_pred - y_true)
        se = tf.sort(se, axis=-1)
        N = se.shape[-1]
        
        if N % 2 != 0:
            medse = se[:,int((N - 1) / 2)]
        else:
            medse = 1 / 2 * (se[:,int(N / 2 - 1)] + se[:,int(N / 2)])

        return medse

class MedianAbsoluteError(tf.keras.losses.Loss):
    """ Class for computing the Median Absolute Error (MedAE) for predictions."""
    def __init__(self):
        super(MedianAbsoluteError, self).__init__()

    def call(self, y_true, y_pred):
        """ Parameters
        ----------
        y_true:
            Ground truth values. shape = `[batch_size, d0, .. dN]`, except sparse loss functions such as sparse categorical crossentropy where shape = `[batch_size, d0, .. dN-1]`
        y_pred:
            The predicted values. shape = `[batch_size, d0, .. dN]`
        """
        se = tf.math.abs(y_pred - y_true)
        se = tf.sort(se, axis=-1)
        N = se.shape[-1]
        
        if N % 2 != 0:
            medse = se[:,int((N - 1) / 2)]
        else:
            medse = 1 / 2 * (se[:,int(N / 2 - 1)] + se[:,int(N / 2)])

        return medse
        
class AlternativeMedianSquaredError(tf.keras.losses.Loss):
    """ Class for computing the Alternative Median Squared Error (AMedSE) for predictions."""
    def __init__(self):
        super(AlternativeMedianSquaredError, self).__init__()

    def call(self, y_true, y_pred):
        """ Parameters
        ----------
        y_true:
            Ground truth values. shape = `[batch_size, d0, .. dN]`, except sparse loss functions such as sparse categorical crossentropy where shape = `[batch_size, d0, .. dN-1]`
        y_pred:
            The predicted values. shape = `[batch_size, d0, .. dN]`
        """
        se = tf.math.square(y_pred - y_true)
        se = tf.sort(se, axis=0)
        N = se.shape[0]
        
        if N % 2 != 0:
            medse = se[int((N - 1) / 2)]
        else:
            medse = 1 / 2 * (se[int(N / 2 - 1)] + se[int(N / 2)])

        return tf.math.reduce_euclidean_norm(medse)

class AlternativeMedianAbsoluteError(tf.keras.losses.Loss):
    """ Class for computing the Alternative Median Absolute Error (AMedAE) for predictions."""
    def __init__(self):
        super(AlternativeMedianAbsoluteError, self).__init__()

    def call(self, y_true, y_pred):
        """ Parameters
        ----------
        y_true:
            Ground truth values. shape = `[batch_size, d0, .. dN]`, except sparse loss functions such as sparse categorical crossentropy where shape = `[batch_size, d0, .. dN-1]`
        y_pred:
            The predicted values. shape = `[batch_size, d0, .. dN]`
        """
        se = tf.math.abs(y_pred - y_true)
        se = tf.sort(se, axis=0)
        N = se.shape[0]
        
        if N % 2 != 0:
            medse = se[int((N - 1) / 2)]
        else:
            medse = 1 / 2 * (se[int(N / 2 - 1)] + se[int(N / 2)])

        return tf.math.reduce_euclidean_norm(medse)

class AlternativeMeanSquaredError(tf.keras.losses.Loss):
    """ Class for computing the Alternative Mean Squared Error (AMSE) for predictions."""
    def __init__(self):
        super(AlternativeMeanSquaredError, self).__init__()

    def call(self, y_true, y_pred):
        """ Parameters
        ----------
        y_true:
            Ground truth values. shape = `[batch_size, d0, .. dN]`, except sparse loss functions such as sparse categorical crossentropy where shape = `[batch_size, d0, .. dN-1]`
        y_pred:
            The predicted values. shape = `[batch_size, d0, .. dN]`
        """
        return tf.math.reduce_euclidean_norm(tf.reduce_mean(tf.math.square(y_pred - y_true), axis=0))

class AlternativeMeanAbsoluteError(tf.keras.losses.Loss):
    """ Class for computing the Alternative Median Absolute Error (AMedAE) for predictions."""
    def __init__(self):
        super(AlternativeMeanAbsoluteError, self).__init__()

    def call(self, y_true, y_pred):
        """ Parameters
        ----------
        y_true:
            Ground truth values. shape = `[batch_size, d0, .. dN]`, except sparse loss functions such as sparse categorical crossentropy where shape = `[batch_size, d0, .. dN-1]`
        y_pred:
            The predicted values. shape = `[batch_size, d0, .. dN]`
        """
        return tf.math.reduce_euclidean_norm(tf.reduce_mean(tf.math.abs(y_pred - y_true), axis=0))
