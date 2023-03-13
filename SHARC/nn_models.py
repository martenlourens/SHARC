import tensorflow as tf

class DenseBlock(tf.keras.Model):
    """ Class constructor of a dense block.

    Parameters
    ----------
    units : int (required)
        Number of units in the Dense layer.
    momentum : float between [0,1], default=0.6 (optional)
        Momentum parameter of the batch normalization layer. Should be close to 1 for slow learning of batch normalization layer. Typically somewhere between 0.6 and 0.85 works fine for big batches.
    alpha : float, default=0.3 (optional)
        Negative slope coefficient of leaky ReLU layer.
    rate : float between [0,1], default=0 (optional)
        Dropout rate.
    
    """
    def __init__(self, units, momentum=0.6, alpha=0.3, rate=0):
        super(DenseBlock, self).__init__()

        # construct a block (NB: ordering of BatchNormalization and activation layers is a topic of debate)
        self.dense = tf.keras.layers.Dense(units, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros')
        self.bn = tf.keras.layers.BatchNormalization(axis=-1, momentum=momentum, epsilon=1e-4) # NB: last axis is the feature axis! small batch: 0.99 (slow learning), big batch: 0.6-0.85
        self.activation = tf.keras.layers.LeakyReLU(alpha=alpha) # alpha specifies the negative slope coefficient
        self.dropout = tf.keras.layers.Dropout(rate=rate) # keep lower than .5

    def call(self, x, training=True):
        x = self.dense(x)
        x = self.bn(x, training=training)
        x = self.activation(x)
        x = self.dropout(x, training=training)

        return x

# NB: this backbone requires the input space to have at least 8 dimensions
class NNPModelBackboneV1(tf.keras.Model):
    """ NNP model backbone class version 1.

    Parameters
    ----------
    D1_units : int (required)
        Number of units in the first dense layer of the network. Should not be less than 4!
    \*\*kwargs : (optional)
        Additional keyword arguments to be passed to each block in this backbone.
    
    """
    def __init__(self, D1_units, **kwargs):
        super(NNPModelBackboneV1, self).__init__()

        self.block1 = DenseBlock(D1_units, **kwargs)
        self.block2 = DenseBlock(D1_units // 2, **kwargs)
        self.block3 = DenseBlock(D1_units // 2 // 2, **kwargs)

    def call(self, inputs, training=True):
        x = self.block1(inputs, training=training)
        x = self.block2(x, training=training)
        x = self.block3(x, training=training)
        
        return x

# NB: this backbone requires the input space to have at least 8 dimensions!
class NNPModelBackboneV2(tf.keras.Model):
    """ NNP model backbone class version 2.

    Parameters
    ----------
    D1_units : int (required)
        Number of units in the first dense layer of the network. Should not be less than 4!
    \*\*kwargs : (optional)
        Additional keyword arguments to be passed to each block in this backbone.
    
    """
    def __init__(self, D1_units, **kwargs):
        super(NNPModelBackboneV2, self).__init__()

        D2_units = int(3 / 4 * D1_units)

        self.block1 = DenseBlock(D1_units, **kwargs)
        self.block2 = DenseBlock(D2_units, **kwargs)
        self.block3 = DenseBlock(D1_units // 2, **kwargs)
        self.block4 = DenseBlock(D2_units // 2, **kwargs)
        self.block5 = DenseBlock(D1_units // 2 // 2, **kwargs)

    def call(self, inputs, training=True):
        x = self.block1(inputs, training=training)
        x = self.block2(x, training=training)
        x = self.block3(x, training=training)
        x = self.block4(x, training=training)
        x = self.block5(x, training=training)
        
        return x

def construct_NNPModel(num_input_features, output_dimensions=2, output_activation="sigmoid", version=2, **kwargs):
    """ Function to construct a NNP (neural network projection) model.

    Parameters
    ----------
    num_input_features : int
        The number of input features.
    output_dimensions : int, default=2 (optional)
        The number of output dimensions of the projection.
    output_activation : str or function, default="sigmoid" (optional)
        Activation function to use.
    version : int, default=2 (optional)
        Version of the NNP model backbone to use.
    \*\*kwargs : (optional)
        Additional keyword arguments will be passed to the NNP model backbone.

    Returns
    -------
    model : tensorflow Model
        A `tf.keras.Model` instance.
    
    """
    inputs = tf.keras.Input(shape=(num_input_features), dtype=tf.float32)

    if version == 1:
        backbone = NNPModelBackboneV1(**kwargs)(inputs) # NB: additional kwargs are passed to backbone model
    elif version == 2:
        backbone = NNPModelBackboneV2(**kwargs)(inputs) # NB: additional kwargs are passed to backbone model

    # NB: depending on the activation of this layer apply some scaling to the embedding data
    outputs = tf.keras.layers.Dense(output_dimensions, use_bias=True, kernel_initializer='glorot_uniform',
                                    bias_initializer='zeros', activation=output_activation)(backbone)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name="NNPModel")

if __name__ == "__main__":
    num_input_features = 20
    model = construct_NNPModel(num_input_features, D1_units=num_input_features)
    print(model.summary())
    # tf.keras.utils.plot_model(model)