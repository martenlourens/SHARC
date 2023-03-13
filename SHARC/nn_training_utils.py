import os
import time
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split

def train_nnp(X, Y_true, model, loss_function, optimizer, labels=None, epochs=10, validation_ratio=0.25, save_path="./NNP", verbose=False):
    """ Function that handles the training of the NNP model.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Feature space training dataset.
    Y_true : array-like, shape (n_samples, n_embedding_dimensions)
        Projection space training dataset.
    model : tensorflow Model
        The `tf.keras.Model` instance to train.
    loss_function :
        A Tensorflow compatible loss function, i.e. it supports auto differentiation, to use for optimization.
    optimizer : tensorflow optimizer
        The `tf.keras.optimizer` to use for optimization.
    labels : array-like, shape (n_samples,), default = None
        An array containing the labels (as numeric values) corresponding to each sample in `X` and `Y_true`.
        When provided it is used to stratify the cross validation set.
    epochs : int, default = 10 (optional)
        Maximum number of epochs.
    validation_ratio : float, default = 0.25 (optional)
        Fraction of the dataset to use for cross validation at each training epoch.
    save_path : str, default = "./NNP" (optional)
        Path the save the checkpoints, training history and trained model to.
    verbose : bool, default = False (optional)
        Controls the verbosity.

    Returns
    -------
    train_loss : numpy.ndarray, shape (epochs,)
        Training loss at each epoch.
    valid_loss : numpy.ndarray, shape (epochs,)
        Validatation loss at each epoch.
    pred_train_loss : numpy.ndarray, shape (epochs,)
        Inferential training loss at each epoch.
    """

    # check if the save_path exists if not create the necessary directories
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    tmp_path = os.path.join(save_path, "tmp")
    if not os.path.isdir(tmp_path):
        os.mkdir(tmp_path)

    checkpoint_prefix = os.path.join(tmp_path, "ckpt")
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)

    @tf.function
    def train_step(X, Y_true, model, loss_function, optimizer):
        with tf.GradientTape() as tape:
            Y_pred = model(X, training=True)

            loss = loss_function(Y_true, Y_pred)

        gradients = tape.gradient(loss, model.trainable_variables)

        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        return loss

    train_loss = []
    valid_loss = []
    pred_train_loss = []

    for epoch in range(epochs):
        start = time.time()

        # construct training and validation sets
        X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y_true, test_size=validation_ratio, shuffle=True, stratify=labels)

        # TODO: add training in batches if model overfits too fast!
        loss = train_step(X_train, Y_train, model, loss_function, optimizer)
        Y_pred = model(X_valid, training=False).numpy()
        val_loss = loss_function(Y_valid, Y_pred).numpy()

        # loss evaluated on training set with training=False (i.e. during inference)
        Y_pred = model(X_train, training=False).numpy()
        pred_loss = loss_function(Y_train, Y_pred).numpy()
        
        train_loss += [loss]
        valid_loss += [val_loss]
        pred_train_loss += [pred_loss]

        # save model every 100 epochs and print training info
        if (epoch + 1) % 100 == 0:
            ckpt_path = checkpoint.save(file_prefix=checkpoint_prefix)
            np.savez_compressed(os.path.join(tmp_path, os.path.split(ckpt_path)[1].split(".")[0] + ".training_history.npz"), train_loss=train_loss, valid_loss=valid_loss, pred_train_loss=pred_train_loss)

            if verbose: print("Time for epoch {} is {} sec. Loss: {:.6f}, Inference loss: {:.6f}, Validation loss: {:.6f}".format(epoch+1, time.time() - start, loss, pred_loss, val_loss))

    # save the trained model
    model.save(os.path.join(save_path, model.name))

    # save losses and validation losses to file
    np.savez_compressed(os.path.join(save_path, model.name, "training_history.npz"), train_loss=train_loss, valid_loss=valid_loss, pred_train_loss=pred_train_loss)

    return train_loss, valid_loss, pred_train_loss