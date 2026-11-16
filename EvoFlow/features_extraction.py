"""Graph features used to predict GA hyper-parameters with the GAM model.

The four features produced by :class:`NetworkFeatures` describe how hard the
node-removal search problem is: the size of the search space
(``log_search_space``), the degree-distribution exponent (``alpha``), the graph
density and the average clustering coefficient.
"""

import networkx as nx
import numpy as np
from scipy.special import gammaln


class NetworkFeatures:
    """Compute GAM input features for a network.

    Attributes:
        G (networkx.DiGraph): Graph the features are computed on.
        b (int): Budget, i.e. the maximum number of nodes the GA may remove.
        n (int): Number of nodes in ``G``.
        e (int): Number of edges in ``G``.
    """

    def __init__(self, G: nx.DiGraph = None, b: int = None):
        """Store the graph and budget used for feature extraction.

        Args:
            G: Directed graph (typically the active subgraph).
            b: Maximum number of removable nodes.
        """
        self.G = G
        self.b = b
        self.n = self.G.number_of_nodes()
        self.e = self.G.number_of_edges()

    def get_log_search_space(self):
        """Calculate log10 of (n choose b) using the log-gamma function.

        Returns:
            float: Base-10 logarithm of the number of candidate removal sets,
            or ``0.0`` when the budget is zero or covers the whole graph.
        """
        if self.b == 0 or self.b >= self.n:
            return 0.0

        log_nck = (
            gammaln(self.n + 1) - gammaln(self.b + 1) - gammaln(self.n - self.b + 1)
        )
        return log_nck / np.log(10)

    def calculate_density(self):
        """Calculate graph density, ``E / (N * (N - 1))``.

        Returns:
            float: Directed-graph density in ``[0, 1]``, or ``0.0`` for graphs
            with fewer than two nodes.
        """
        if self.n <= 1:
            return 0.0
        return self.e / (self.n * (self.n - 1))

    def calculate_clustering_coefficient(self):
        """Calculate the average clustering coefficient of the undirected graph.

        Returns:
            float: Average clustering coefficient, or ``0.0`` for graphs with
            fewer than four nodes.
        """
        if self.n < 4:
            return 0.0

        undirected = self.G.to_undirected()
        return nx.average_clustering(undirected)

    def estimate_alpha(self):
        """Estimate the power-law exponent alpha of the degree distribution.

        Uses a maximum-likelihood estimate on the (shifted) degree sequence.

        Returns:
            float: Rounded exponent ``alpha``, or ``0`` when there are too few
            non-isolated nodes or the estimate is outside ``[1.5, 10]``.
        """
        degrees = [d for n, d in self.G.degree() if d > 0]

        if len(degrees) < 10:
            return 0

        degrees = np.array(degrees, dtype=float)
        d_min = degrees.min()

        if d_min < 1:
            d_min = 1

        degrees_adjusted = degrees - d_min + 1

        n = len(degrees_adjusted)

        try:
            alpha = 1 + n / np.sum(np.log(degrees_adjusted / d_min))

            if alpha < 1.5 or alpha > 10:
                return 0

            return round(alpha, 3)
        except (ZeroDivisionError, RuntimeWarning):
            return 0

    def extract_features(self):
        """Extract all features consumed by the GAM in model order.

        Returns:
            dict: Mapping ``feature name -> [value]`` with the keys
            ``log_search_space``, ``alpha``, ``density`` and
            ``clustering_coef`` (each value is a one-element list so the result
            can be passed directly to :class:`pandas.DataFrame`).
        """
        features = {
            "log_search_space": [self.get_log_search_space()],
            "alpha": [self.estimate_alpha()],
            "density": [self.calculate_density()],
            "clustering_coef": [self.calculate_clustering_coefficient()],
        }
        return features
