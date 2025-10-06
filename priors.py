import numpy as np


class FHNMultivariateUniform:
    def __init__(self, bounds, second_upper):
        """
        bounds: list of (a, b) for each dimension except the second
        second_upper: upper bound for the second variable (user input)
        """
        self.bounds = np.array(bounds, dtype=float)
        self.second_upper = second_upper
        self.dim = len(bounds) + 1

    def rvs(self, size=1, random_state=None):
        rng = np.random.default_rng(random_state)
        # Sample first variable
        epsilon = rng.uniform(low=self.bounds[0, 0], high=self.bounds[0, 1], size=size)
        # Sample second variable conditional on epsilon
        second_low = epsilon / 4
        second_high = np.full(size, self.second_upper)
        second = rng.uniform(low=second_low, high=second_high)
        # Sample remaining variables
        others = np.column_stack([
            rng.uniform(low=a, high=b, size=size)
            for (a, b) in self.bounds[1:]
        ])
        samples = np.column_stack([epsilon, second, others])
        return samples

    def pdf(self, x):
        x = np.atleast_2d(x)
        epsilon = x[:, 0]
        second = x[:, 1]
        # Check bounds for first variable
        cond1 = (epsilon >= self.bounds[0, 0]) & (epsilon <= self.bounds[0, 1])
        # Check bounds for second variable
        cond2 = (second >= epsilon / 4) & (second <= self.second_upper)
        # Check bounds for remaining variables
        cond3 = np.ones(x.shape[0], dtype=bool)
        for i, (a, b) in enumerate(self.bounds[1:], start=2):
            cond3 &= (x[:, i] >= a) & (x[:, i] <= b)
        # Compute density
        vol_epsilon = self.bounds[0, 1] - self.bounds[0, 0]
        vol_second = self.second_upper - (epsilon / 4)
        vol_others = np.prod([b - a for (a, b) in self.bounds[1:]])
        density = 1.0 / (vol_epsilon * vol_second * vol_others)
        inside = cond1 & cond2 & cond3
        return np.where(inside, density, 0.0)

    @property
    def support(self):
        # Support for second variable depends on epsilon
        return self.bounds, self.second_upper


class UniformND:
    """Independent uniform prior over an N-dimensional box.

    bounds = [(low_1, high_1), ..., (low_D, high_D)]
    """
    def __init__(self, bounds):
        bounds = np.asarray(bounds, dtype=float)
        if bounds.ndim != 2 or bounds.shape[1] != 2:
            raise ValueError("Expected bounds shape (D,2) for D parameters")
        self.bounds = bounds
        self.lows = bounds[:, 0]
        self.highs = bounds[:, 1]
        if np.any(self.highs <= self.lows):
            raise ValueError("Each upper bound must be > lower bound")
        self.dim = bounds.shape[0]
        self.volume = float(np.prod(self.highs - self.lows))

    def rvs(self, size=1, random_state=None):
        rng = np.random.default_rng(random_state)
        return rng.uniform(self.lows, self.highs, size=(size, self.dim)).astype(float)

    def pdf(self, x):
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            if x.shape[0] != self.dim:
                raise ValueError(f"x must have {self.dim} elements")
            inside = np.all((x >= self.lows) & (x <= self.highs))
            return (1.0 / self.volume) if inside else 0.0
        elif x.ndim == 2:
            if x.shape[1] != self.dim:
                raise ValueError(f"x must have shape (n, {self.dim})")
            inside = np.all((x >= self.lows) & (x <= self.highs), axis=1)
            out = np.zeros(x.shape[0], dtype=float)
            out[inside] = 1.0 / self.volume
            return out
        else:
            raise ValueError("x must be a 1D or 2D array")

    @property
    def support(self):
        return self.bounds


__all__ = ["FHNMultivariateUniform", "UniformND"]
