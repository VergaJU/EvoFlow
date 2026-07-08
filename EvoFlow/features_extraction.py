import numpy as np
import networkx as nx
from scipy.special import gammaln


class NetworkFeatures:
    def __init__(self, G: nx.DiGraph = None, b: int = None):
        self.G = G
        self.b = b
        self.n = self.G.number_of_nodes()
        self.e = self.G.number_of_edges()

    def get_log_search_space(self):
        """Calculate log10 of (n choose b) using log-gamma function."""
        if self.b == 0 or self.b >= self.n:
            return 0.0

        log_nck = (
            gammaln(self.n + 1) - gammaln(self.b + 1) - gammaln(self.n - self.b + 1)
        )
        return log_nck / np.log(10)

    def calculate_density(self):
        """Calculate graph density: E / (N * (N-1))."""
        if self.n <= 1:
            return 0.0
        return self.e / (self.n * (self.n - 1))

    def calculate_clustering_coefficient(self):
        """Calculate average clustering coefficient."""
        if self.n < 4:
            return 0.0

        undirected = self.G.to_undirected()
        return nx.average_clustering(undirected)

    def estimate_alpha(self):
        """Estimate power-law exponent alpha from degree distribution.
        Uses maximum likelihood estimation for power-law distribution.
        Returns 0 if fitting fails.
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
        """Extract all graph features for ML model.

        Args:
            G: NetworkX graph
            budget: Number of nodes to remove (int)

        Returns:
            dict: Feature dictionary
        """
        features = {
            "log_search_space": [self.get_log_search_space()],
            "alpha": [self.estimate_alpha()],
            "density": [self.calculate_density()],
            "clustering_coef": [self.calculate_clustering_coefficient()],
        }
        return features
