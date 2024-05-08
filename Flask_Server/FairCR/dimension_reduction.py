import math

from sklearn.manifold import TSNE
import numpy as np

def euclidean_similiarty(datapoint1, datapoint2):
    """

    Parameters
    ----------
    datapoint1 an array containing values for the attributes
    datapoint2 an array containing values for the atteibutes

    Returns the euclidean distance
    -------

    """
    sum = 0
    for index in range(len(datapoint1)):
        sum +=  (datapoint1[index] - datapoint2[index])**2
    return math.sqrt(sum)

def run_tsne(data, n_components=2, perplexity=30, n_iter=400, random_state=0):
    """
    Runs the t-SNE (t-Distributed Stochastic Neighbor Embedding) algorithm on the given data and returns the embedded data points.

    Parameters:
    -----------
    data : numpy array-like
        The input data to be used for t-SNE dimension reduction.

    n_components : int, optional (default=2)
        The number of dimensions in the embedded space.

    perplexity : float, optional (default=30)
        A hyperparameter that influences the balance between local and global neighborhood relationships.

    n_iter : int, optional (default=300)
        The number of iterations the algorithm should perform.

    random_state : int, optional (default=0)
        The random seed for result reproducibility.

    Returns:
    --------
    numpy array
        The embedded data points in the reduced space.

    """

    # Initialize the t-SNE algorithm
    tsne = TSNE(n_components=n_components, perplexity=perplexity, n_iter=n_iter, random_state=random_state)

    # Apply t-SNE to the data
    embedded_data = tsne.fit_transform(data)

    # Convert the data in the right (Charts.js) expected form
    datapoint_list = embedded_data.tolist()

    return datapoint_list

