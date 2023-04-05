import sys
import numpy as np
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.stats import spearmanr
from sklearn.utils.estimator_checks import check_is_fitted
from sklearn.exceptions import NotFittedError

class Metrics(object):
    """ A base class for the computation of some basic metrics that quantify the performance of DR algorithms.

    Parameters
    ----------
    metric : str or list, default=[\"euclidean\", \"euclidean\"]
        Metrics to use when computing distances in the feature space and the projection space.
        When a string is provided that same metric will be used for both the feature space and the projection space.
        Values are passed to `scipy.spatial.distance.pdist`.
    k : int, default=7
        Number of nearest neighbors to consider when computing the various metrics.
        Used by `metric_trustworthiness`, `metric_continuity`, `metric_jaccard_similarity_coefficient`, `metric_neighborhood_hit` and `metric_distribution_consistency`.
    
    """
    def __init__(self, metric=["euclidean", "euclidean"], k = 7):
        # configuration parameters
        # check metric
        if type(metric) == str:
            self._metric = 2 * [metric]
        else:
            self._metric = metric

        self.k = k

        # metric cache
        # local neighborhood metrics
        self._M_trustworthiness = None
        self._M_continuity = None
        self._M_jaccard_similarity_coefficient = None

        # distance preservation metrics
        self._M_normalized_stress = None
        self._M_shepard_goodness = None

        # purity/VSC metrics
        self._M_neighborhood_hit = None
        self._M_distance_consistency = None
        self._M_distribution_consistency = None

        # variable storing composite metric
        self._M_total = None

        return self

    def __delete__(self):
        try:
            check_is_fitted(self, ['_N', '_Y', '_labels', '_dist_X', '_dist_Y', '_nn_X', '_nn_Y'])
            
            # free memory
            del self._Y
            if self._labels is not None:
                del self._labels
            del self._dist_X
            del self._dist_Y
            del self._nn_X
            del self._nn_Y

        except NotFittedError:
            pass

    def fit(self, X, Y, labels=None):
        """ Fit the provided data to the metric instance.
        That is, for both `X` and `Y` compact distance matrices and nearest neighbor sets are computed.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Feature space dataset.
        Y : array-like, shape (n_samples, n_embedding_dimensions)
            Projection space dataset.
        labels : array-like, shape (n_samples, ), default=None
            An array of label values for each sample. Only required for purity/VSC metrics such as `metric_neighborhood_hit`, `metric_distance_consistency` and `metric_distribution_consistency`

        Returns
        -------
        self : object
            Returns self.
        
        """
        # check shapes
        if X.shape[0] != Y.shape[0]:
            raise ValueError("Number of samples in Y does not match the number of samples in X! X has %d samples whereas Y has %d samples." % (X.shape[0], Y.shape[0]))
        self._N = X.shape[0]

        self._Y = Y # stored for distance consistency metric

        # check shape of labels
        if labels is not None and labels.shape[0] != self._N:
            raise ValueError("Number of samples in labels does not match the number of samples in X! X has %d samples whereas labels has %d samples." % (self._N, labels.shape[0]))
        self._labels = labels

        # compute distance matrices (NB: these are stored as float32 to save resources)
        dist = pdist(X, metric = self._metric[0])
        dist = dist.astype(np.float32)
        self._dist_X = squareform(dist, force='tomatrix')

        dist = pdist(Y, metric = self._metric[1])
        dist = dist.astype(np.float32)
        self._dist_Y = squareform(dist, force='tomatrix')

        del dist

        # indices that sort rows in distance matrices (NB: these are stored as uint16 to save resources)
        self._nn_X = np.argsort(self._dist_X, kind="mergesort", axis=1).astype(np.uint16)
        self._nn_Y = np.argsort(self._dist_Y, kind="mergesort", axis=1).astype(np.uint16)

        # reset metric cache
        # local neighborhood metrics
        self._M_trustworthiness = None
        self._M_continuity = None
        self._M_jaccard_similarity_coefficient = None

        # distance preservation metrics
        self._M_normalized_stress = None
        self._M_shepard_goodness = None

        # purity/VSC metrics
        self._M_neighborhood_hit = None
        self._M_distance_consistency = None
        self._M_distribution_consistency = None

        # variable storing composite metric
        self._M_total = None

        return self

    # trustworthiness metric
    def metric_trustworthiness(self):
        r""" Function to compute the trustworthiness metric which quantifies the **proportion of false neighbors** in the projection.
        The functional definition reads as follows:

        .. math::
            M_t(k) = 1 - \frac{2}{Nk(2N-3k-1)}\sum^{N}_{i=1}\sum_{j\in \mathcal{U}_i^k}(r(i,j) - k)
            :label: trustworthiness

        In this definition, :math:`N` is the number of samples in the dataset and :math:`k` is the number of nearest neighbors 
        to consider and should always be smaller than :math:`N / 2` for the metric to be properly normalized.
        The set :math:`\mathcal{U}_i^k` consists of the :math:`k` nearest neighbors of sample :math:`i` in the projection that are **not** 
        amongst the :math:`k` nearest neighbors of :math:`i` in the original space. The quantity :math:`r(i,j)` specifies the rank of 
        the point :math:`j` when feature vectors are ordered based on their distance to point :math:`i` in the original space.
        
        Returns
        -------
        trustworthiness : float
            The value between :math:`[0,1]` yielded by the trustworthiness metric.
        
        """
        # check cache
        if self._M_trustworthiness is not None:
            return self._M_trustworthiness

        check_is_fitted(self, ['_N', '_nn_X', '_nn_Y'])

        # check k < N / 2 (required by normalization factor in trustworthiness definition)
        if self.k >= self._N / 2:
            raise ValueError("k should be smaller than N/2 where N in the number of samples in each dataset!")
        norm = 2 / (self._N * self.k * (2 * self._N - 3 * self.k - 1))

        Z = 0
        for i in range(self._N):
            Z += np.sum([max(0, np.where(self._nn_X[i] == j)[0][0] - self.k) for j in self._nn_Y[i][1:self.k+1]])

        self._M_trustworthiness = 1 - norm * Z
        return self._M_trustworthiness

    # continuity metric
    def metric_continuity(self):
        r""" Function to compute the continuity metric which quantifies the **proportion of missing neighbors** in the projection.
        The functional definition reads as follows:

        .. math::
            M_c(k) = 1 - \frac{2}{Nk(2N-3k-1)}\sum^{N}_{i=1}\sum_{j\in \mathcal{V}^k_i}(\hat{r}(i,j)-k)
            :label: continuity

        In this definition, :math:`N` is the number of samples in the dataset and :math:`k` is the number of nearest neighbors 
        to consider and should always be smaller than :math:`N / 2` for the metric to be properly normalized.
        The set :math:`\mathcal{V}^{k}_i` consists of the :math:`k` nearest neighbors of sample :math:`i` in original data space 
        that are not among the :math:`k` data vectors after the projection. The quantity :math:`\hat{r}(i,j)` specifies the rank of 
        the point :math:`j` when feature vectors are based on their distance to point :math:`i` after the projection.

        Returns
        -------
        continuity : float
            The value between :math:`[0,1]` yielded by the continuity metric.
        
        """

        # continuity : float, between :math:`\int_0^1 dx`
        #     The value yielded by the continuity metric.
        
        # check cache
        if self._M_continuity is not None:
            return self._M_continuity

        check_is_fitted(self, ['_N', '_nn_X', '_nn_Y'])

        # check k < N / 2 (required by normalization factor in continuity definition)
        if self.k >= self._N / 2:
            raise ValueError("k should be smaller than N/2 where N in the number of samples in each dataset!")
        norm = 2 / (self._N * self.k * (2 * self._N - 3 * self.k - 1))

        Z = 0
        for i in range(self._N):
            Z += np.sum([max(0, np.where(self._nn_Y[i] == j)[0][0] - self.k) for j in self._nn_X[i][1:self.k+1]])

        self._M_continuity = 1 - norm * Z
        return self._M_continuity

    # Jaccard similarity coefficient metric
    def metric_jaccard_similarity_coefficient(self):
        r""" Function to compute the Jaccard similarity coefficient metric which quantifies the **proportion of overlap** between the :math:`k`-nearest 
        neighbor sets in the feature space and the projection space. The functional definition reads as follows:

        .. math::
            M_J(k) = \frac{1}{N}\sum^{N}_{i=1}\frac{\left|\mathcal{N}^k_i \cap \mathcal{M}^k_i\right|}{\left|\mathcal{N}^k_i \cup \mathcal{M}^k_i\right|}
            :label: jaccard
        
        In this definition, :math:`N` is the number of samples in the dataset and :math:`k` is the number of nearest neighbors 
        to consider. The set :math:`\mathcal{N}^{k}_i` consists of the :math:`k` nearest neighbors of sample :math:`i` in original 
        data space. The set :math:`\mathcal{M}^{k}_i` consists of the :math:`k` nearest neighbors of sample :math:`i` in the projection.

        Returns
        -------
        jaccard_similarity_coefficient : float
            The value between :math:`[0,1]` yielded by the Jaccard similarity coefficient metric.
        
        """
        # check cache
        if self._M_jaccard_similarity_coefficient is not None:
            return self._M_jaccard_similarity_coefficient

        check_is_fitted(self, ['_N', '_nn_X', '_nn_Y'])

        Z = 0
        for i in range(self._N):
            Z += np.intersect1d(self._nn_Y[i][1:self.k+1], self._nn_X[i][1:self.k+1]).size / np.union1d(self._nn_Y[i][1:self.k+1], self._nn_X[i][1:self.k+1]).size

        self._M_jaccard_similarity_coefficient = Z / self._N
        return self._M_jaccard_similarity_coefficient

    # normalized stress metric
    def metric_normalized_stress(self):
        r""" Function to compute the normalized stress metric which quantifies the respective **mismatch** between pointwise distances in the 
        feature space and the projection space. The functional definition reads as follows:

        .. math::
            M_{\sigma}(k) = \frac{\sum^{N}_{i=1}\sum^{N}_{j=1}\left(\Delta^n(\mathbf{x}_i,\mathbf{x}_j)-\Delta^m(P\left(\mathbf{x}_i\right),P\left(\mathbf{x}_j)\right)\right)^2}{\sum^{N}_{i=1}\sum^{N}_{j=1}\Delta^n(\mathbf{x}_i,\mathbf{x}_j)^2}
            :label: normalized_stress

        In this definition, :math:`N` is the number of samples in the dataset. The function :math:`\Delta^n(\mathbf{x}_i, \mathbf{x}_j)` returns
        the distance between points :math:`i` and :math:`j` in :math:`n`-dimensions.

        Returns
        -------
        normalized_stress : float
            The value between :math:`[0, \infty]` yielded by the normalized stress metric.
        
        """
        # check cache
        if self._M_normalized_stress is not None:
            return self._M_normalized_stress

        check_is_fitted(self, ['_dist_X', '_dist_Y'])

        dist_X = squareform(self._dist_X, force='tovector')
        dist_Y = squareform(self._dist_Y, force='tovector')

        Z0 = np.sum((dist_X - dist_Y) ** 2)
        Z1 = np.sum(dist_X ** 2)

        self._M_normalized_stress = Z0 / Z1
        return self._M_normalized_stress

    # shepard diagram
    def shepard_diagram(self):
        """ Function that returns the Shepard diagram.

        Returns
        -------
        shepard_diagram : array-like (n_pairs, 2)
            An array of pairwise distances between points in the original data space and the projection.
        
        """
        check_is_fitted(self, ['_dist_X', '_dist_Y'])
        
        dist_X = squareform(self._dist_X, force='tovector')
        dist_Y = squareform(self._dist_Y, force='tovector')

        return np.column_stack((dist_X, dist_Y))

    # spearman rank correlation metric of shepard diagram
    def metric_shepard_goodness(self, return_shepard=False):
        r""" Function that computes the Shepard goodness metric, i.e. the spearman rank correlation of the Shepard diagram.

        Parameters
        ----------
        return_shepard : bool, default=False
            Controls whether to return the Shepard diagram as well.
        
        Returns
        -------
        shepard_goodness : float
            The value between :math:`[0,1]` of the Shepard goodness metric.
        
        """
        # check cache
        if self._M_shepard_goodness is not None and not return_shepard:
            return self._M_shepard_goodness

        SD = self.shepard_diagram()

        self._M_shepard_goodness = spearmanr(SD[:,0], SD[:,1])[0]

        # check for negative correlation
        if self._M_shepard_goodness < 0:
            print(f"[warning] Negative correlation in Shepard diagram! Spearman rank is {self._M_shepard_goodness}.")

        if return_shepard:
            return self._M_shepard_goodness, SD
        return self._M_shepard_goodness

    # neighborhood hit metric
    def metric_neighborhood_hit(self):
        r""" Function to compute the neighborhood hit metric which measures how well separated datapoints with different labels are in the projection.
        The functional definition reads as follows:

        .. math::
            M_{NH}(k) = \frac{1}{kN}\sum^{N}_{i=1}\left|\left\{j\in\mathcal{N}^{k}_{i} | l_j = l_i\right\}\right|
            :label: neighborhood_hit

        In this definition, :math:`N` is the number of samples in the dataset and :math:`k` is the number of nearest neighbors 
        to consider. The set :math:`\mathcal{N}^k_i` is the set of nearest neighbors of point :math:`i` in the projection space and :math:`l_i` denotes the 
        label of a point :math:`i`.

        Returns
        -------
        normalized_stress : float
            The value between :math:`[0, \infty]` yielded by the normalized stress metric.
        
        """
        # check cache
        if self._M_neighborhood_hit is not None:
            return self._M_neighborhood_hit

        check_is_fitted(self, ['_N', '_labels', '_nn_Y'])

        # verify labels were specified at fit
        if self._labels is None:
            raise ValueError("The neighborhood hit metric requires labels to be set using the `fit` method!")

        Z = 0
        for i in range(self._N):
            Z += np.sum(self._labels[i] == self._labels[self._nn_Y[i][1:self.k+1]])

        self._M_neighborhood_hit = Z / (self.k * self._N)
        return self._M_neighborhood_hit

    # Distance Consistency (DSC) metric (Sips et al. (2009))
    def metric_distance_consistency(self):
        r""" Function to compute the distance consistency metric which measures how well separated data clusters with different labels are in the projection.
        The functional definition reads as follows:
            
        .. math::
            M_{\text{DSC}} = 1 - \frac{\left|\left\{\vec{x}\in D : \text{CD}(\vec{x}, \text{centr}(\text{clabel}(\vec{x}))) \neq 1\right\}\right|}{N}
            :label: distance_consistency
        
        In this definition, :math:`N` is the number of samples in the dataset :math:`D` and :math:`\text{CD}(\vec{x}, \text{centr}(\text{clabel}(\vec{x})))` 
        is the so-called *centroid distance* which is defined as follows:
            
        .. math::
            \text{CD}(\vec{x}, \text{centr}(\text{clabel}(\vec{x}))) =
            \begin{cases}
                1\quad d(\vec{x},\text{centr}(\text{clabel}(\vec{x}))) < d(\vec{x},\text{centr}(c_i)) \forall i \in [0, m] \wedge c_i \neq \text{clabel}(\vec{x})\\
                0\quad\text{otherwise}
            \end{cases}
        
        where :math:`\text{centr}(c_i)` is the position of the centroid corresponding to all datapoints with class label :math:`c_i`, :math:`\text{clabel}(\vec{x})` gets 
        the class label of datapoint :math:`\vec{x}` and :math:`d(\vec{x},\vec{y})` is the distance between points :math:`\vec{x}` and :math:`\vec{y}`.

        Returns
        -------
        distance_consistency : float
            The value between :math:`[0, 1]` yielded by the distance consistency metric.
        
        """
        # check cache
        if self._M_distance_consistency is not None:
            return self._M_distance_consistency

        check_is_fitted(self, ['_N', '_Y', '_labels'])

        # verify labels were specified at fit
        if self._labels is None:
            raise ValueError("The distance consistency metric requires labels to be set using the `fit` method!")

        # determine the defined unique labels
        known_labels = np.unique(self._labels)

        # for each label compute centroids
        centroids = np.array([np.mean(self._Y[self._labels == l], axis=0) for l in known_labels])

        # compute distances to each centroid for each sample in Y
        distances = cdist(centroids, self._Y, metric=self._metric[1])

        # compute consistency distance for each sample in Y
        consistency_distances = known_labels[np.argmin(distances, axis=0)] == self._labels

        # compute distance consistency metric (DCM)
        self._M_distance_consistency = 1 - np.sum(~consistency_distances) / self._N

        return self._M_distance_consistency

    # Distribution Consistency (DC) metric (using knn)
    def metric_distribution_consistency(self):
        r""" Function to compute the distribution consistency metric which measures how well separated data with different class labels are in the projection.
        The functional definition reads as follows:
            
        .. math::
            M_{\text{DC}} = 1 + \frac{1}{N\log_2(m)}\sum_{\vec{x}\in D}\sum_{i=0}^{m}\frac{p_{c_i}}{\sum_{i=0}^m p_{c_i}}\log_2\left(\frac{p_{c_i}}{\sum_{i=0}^m p_{c_i}}\right)
            :label: distribution_consistency
        
        In this definition, :math:`N` is the number of samples in the dataset :math:`D`, :math:`m` is the number of unique class labels and :math:`p_{c_i}` is the number 
        of datapoints of class :math:`c_i` in the nearest neighbor set of a point :math:`\vec{x}`. The way this metric is defined, it measures the average purity with respect 
        to the class labels in the neighborhood of all points in the dataset. To probe the purity it uses the Shannon entropy.

        Returns
        -------
        distribution_consistency : float
            The value between :math:`[0, 1]` yielded by the distribution consistency metric.
        
        """
        # check cache
        if self._M_distribution_consistency is not None:
            return self._M_distribution_consistency
        
        check_is_fitted(self, ['_N', '_labels', '_nn_Y'])

        # verify labels were specified at fit
        if self._labels is None:
            raise ValueError("The distribution consistency metric requires labels to be set using the `fit` method!")

        # determine the defined unique labels
        known_labels = np.unique(self._labels)
        m = known_labels.size

        # for each sample compute the entropy in class label among its k nearest neighbors
        H = []
        for i in range(self._N):
            lbls = self._labels[self._nn_Y[i][:self.k+1]]

            # generate histogram of labels
            pc = np.histogram(lbls, bins=m, range=(known_labels[0], known_labels[-1]))[0]

            # compute entropy
            H += [-1 * np.sum(pc /np.sum(pc) * np.log2(pc / np.sum(pc), where=pc>0))]
        H = np.array(H)

        # compute distribution consistency metric
        self._M_distribution_consistency = 1 - 1 / (np.log2(m) * self._N) * np.sum(H)

        return self._M_distribution_consistency

    def get_summary(self):
        """ Function to get a summary of the computed metrics.

        Returns
        -------
        summary : dict
            A dictionary containing all computed metrics and their values.
        
        """
        implemented_metrics = []
        for method in dir(self):
            tmp = method.split('_', 1)
            if tmp[0] == 'metric':
                implemented_metrics.append(tmp[1])

        summary = {}
        for metric in implemented_metrics:
            metric_result = eval(f"self._M_{metric}")
            if metric_result is not None:
                summary[metric] = metric_result

        return summary

    def print_summary(self, file=sys.stdout, end='\n'):
        r""" Function to print a summary of the computed metrics.

        Parameters
        ----------
        file : file-like object (stream), default=sys.stdout

        end : string appended after the last value, default='\\n'
        
        """
        summary = self.get_summary()
        summary_str = ""
        for k, v in summary.items():
            summary_str += "\t{}: {:.6f}\n".format(k, v)
        print("Metric summary:\n", summary_str, file=file, end=end)
        

class DR_MetricsV1(Metrics):
    """ Metric class for DR optimization using a metric composed of the trustworthiness, continuity, neighborhood hit and Shepard goodness metrics.
    Metric functions are inherited from the `metrics.Metrics` class.

    Parameters
    ----------
    metric : str or list, default=[\"euclidean\", \"euclidean\"]
        Metrics to use when computing distances in the feature space and the projection space.
        When a string is provided that same metric will be used for both the feature space and the projection space.
        Values are passed to `scipy.spatial.distance.pdist`.
    k : int, default=7
        Number of nearest neighbors to consider when computing the various metrics.
        Used by `metric_trustworthiness`, `metric_continuity`, `metric_jaccard_similarity_coefficient`, `metric_neighborhood_hit` and `metric_distribution_consistency`.
    
    """
    def __init__(self, metric=["euclidean", "euclidean"], k = 7):
        super().__init__(metric = metric, k = k)

    def __delete__(self):
        super().__delete__()

    # total scalar metric (to be optimized)
    def metric_total(self):
        r""" Function to compute the optimization metric.

        Returns
        -------
        total : float
            The value between :math:`[0, 1]` of the composite metric, i.e.:
            
            .. math::
                \frac{1}{4}\left(\text{trustworthiness} + \text{continuity} + \text{neighborhood hit} + \text{Shepard goodness}\right)
        
        """
        self._M_total = 0

        self._M_total += self.metric_trustworthiness()
        self._M_total += self.metric_continuity()
        self._M_total += self.metric_neighborhood_hit()
        self._M_total += self.metric_shepard_goodness()

        self._M_total /= 4

        return self._M_total
    
class DR_MetricsV2(Metrics):
    """ Metric class for DR optimization using a metric composed of only the distribution consistency metric.
    The metric function for distribution consistency is inherited from the `metrics.Metrics` class.

    Parameters
    ----------
    metric : str or list, default=[\"euclidean\", \"euclidean\"]
        Metrics to use when computing distances in the feature space and the projection space.
        When a string is provided that same metric will be used for both the feature space and the projection space.
        Values are passed to `scipy.spatial.distance.pdist`.
    k : int, default=7
        Number of nearest neighbors to consider when computing the various metrics.
        Used by `metric_trustworthiness`, `metric_continuity`, `metric_jaccard_similarity_coefficient`, `metric_neighborhood_hit` and `metric_distribution_consistency`.
    
    """
    def __init__(self, metric=["euclidean", "euclidean"], k = 7):
        super().__init__(metric = metric, k = k)

    def __delete__(self):
        super().__delete__()

    # total scalar metric (to be optimized)
    def metric_total(self):
        r""" Function to compute the optimization metric.

        Returns
        -------
        total : float
            The value between :math:`[0, 1]` of the optimization metric, i.e. distribution consistency.
        
        """
        self._M_total = self.metric_distribution_consistency()
        return self._M_total

class LGC_Metrics(Metrics):
    """ Metric class for LGC optimization using a metric composed of only the distribution consistency metric.
    The metric function for distribution consistency is inherited from the `metrics.Metrics` class.

    Parameters
    ----------
    metric : str or list, default=[\"euclidean\", \"euclidean\"]
        Metrics to use when computing distances in the feature space and the projection space.
        When a string is provided that same metric will be used for both the feature space and the projection space.
        Values are passed to `scipy.spatial.distance.pdist`.
    k : int, default=7
        Number of nearest neighbors to consider when computing the various metrics.
        Used by `metric_trustworthiness`, `metric_continuity`, `metric_jaccard_similarity_coefficient`, `metric_neighborhood_hit` and `metric_distribution_consistency`.
    
    """
    def __init__(self, metric="euclidean", k = 7):        
        super().__init__(metric = metric, k = k)

    def __delete__(self):
        super().__delete__()

    # total scalar metric (to be optimized)
    def metric_total(self, k = 7):
        r""" Function to compute the optimization metric.

        Returns
        -------
        total : float
            The value between :math:`[0, 1]` of the optimization metric, i.e. distribution consistency.
        
        """
        self._M_total = self.metric_distribution_consistency()
        return self._M_total

# TODO: 
# average local error function for Metrics class