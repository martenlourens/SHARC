import tensorflow as tf
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import check_X_y, check_array
from sklearn.utils.estimator_checks import check_is_fitted, check_estimator
from sklearn.utils.multiclass import unique_labels

import matplotlib.pyplot as plt
from sklearn.inspection import DecisionBoundaryDisplay

from .plot_funcs import plot_projection

class SDRNNPClassifier(ClassifierMixin, BaseEstimator):
    """ A classifier which implements SDR-NNP based classification.

    Parameters
    ----------
    nnp_model_path : str, default=None
        Path to the stored SDR-NNP model (required).
    classifier : object, default=None
        Classifier used for the final classification (required).
        
    Attributes
    ----------
    X\_ : ndarray, shape (n_samples, n_features)
        The input passed during :meth:`fit`.
    y\_ : ndarray, shape (n_samples,)
        The labels passed during :meth:`fit`.
    classes\_ : ndarray, shape (n_classes,)
        The classes seen at :meth:`fit`.
    """
    def __init__(self, nnp_model_path=None, classifier=None):
        self.nnp_model_path = nnp_model_path
        self.classifier = classifier

    def fit(self, X, y):
        """ Fit the SDR-NNP based classifier from the training dataset.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,)
            The target values. An array of int.
            
        Returns
        -------
        self : object
            Returns self.
        """
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)

        self.n_features_in_ = X.shape[1]        

        # Store the classes seen during fit
        self.classes_ = unique_labels(y)

        # Load SDR-NNP model
        self.sdr_nnp_model_ = tf.keras.models.load_model(self.nnp_model_path)

        # Apply SDR-NNP to X
        self.X_ = X
        self.Y_ = self.sdr_nnp_model_(X, training=False).numpy()
        self.y_ = y

        # Fit the classifier
        self.classifier.fit(self.Y_, self.y_)

        # Return the classifier
        return self

    def predict(self, X):
        """ Predict the class labels for the provided data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y : ndarray, shape (n_samples,)
            Class labels for each data sample.
        """
        # Check if fit has been called
        check_is_fitted(self, ['X_', 'Y_', 'y_', 'sdr_nnp_model_'])

        # Input validation
        X = check_array(X)

        # Apply SDR-NNP to X
        Y = self.sdr_nnp_model_(X, training=False).numpy()

        return self.classifier.predict(Y)

    def predict_proba(self, X):
        """ Return probability estimates for the test data X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y : ndarray, shape (n_samples,)
            Class labels for each data sample.
        """
        # Check if fit has been called
        check_is_fitted(self, ['X_', 'Y_', 'y_', 'sdr_nnp_model_'])

        # Input validation
        X = check_array(X)

        # Apply SDR-NNP to X
        Y = self.sdr_nnp_model_(X, training=False).numpy()

        return self.classifier.predict_proba(Y)

    def plot_projection(self, X, y=None, ax=None):
        """ Plot the SDR-NNP projection of the input data X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples.
        y : array-like, shape (n_samples,), default=None
            The target values. An array of int.
        ax : matplotlib Axes, default=None
            Axes object to plot on. If `None`, a new figure and axes is created.

        Returns
        -------
        ax : matplotlib Axes
            Axes object that was plotted on.
        """
        # Check if fit has been called
        check_is_fitted(self, ['X_', 'Y_', 'y_', 'sdr_nnp_model_'])

        # Input validation
        X = check_array(X)

        # Apply SDR-NNP to X
        Y = self.sdr_nnp_model_(X, training=False).numpy()

        # Plot the SDR-NNP projection
        ax = plot_projection(Y, labels=y, ax=ax)

        return ax

    def plot_classifier_decision_boundaries(self, ax=None, grid_resolution=200, eps=0.2, plot_method="contourf", cmap=plt.cm.RdYlBu, alpha=0.3, **kwargs):
        """ Plot decision boundaries for the trained classifier.

        Parameters
        ----------
        ax : matplotlib Axes, default=None
            Axes object to plot on. If `None`, a new figure and axes is created.
        \*\*kwargs : 
            Additional arguments are passed to `sklearn.inspection.DecisionBoundaryDisplay.from_estimator()`.

        Returns
        -------
        display : DecisionBoundaryDisplay
            Object storing the result.
        """
        # Check if fit has been called
        check_is_fitted(self, ['X_', 'Y_', 'y_', 'sdr_nnp_model_'])

        display = DecisionBoundaryDisplay.from_estimator(self.classifier, self.Y_, grid_resolution=grid_resolution, eps=eps, plot_method=plot_method, cmap=cmap, ax=ax, alpha=alpha)
        return display