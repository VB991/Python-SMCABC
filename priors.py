"""Parameter prior distributions used by the ABC/SMC code.

This module centralizes the prior classes that were previously
defined in ``main.py``.
"""

import numpy as np


class MultivariateUniform:
    """Uniform prior with a conditional second dimension.

    - ``bounds``: list of ``(a, b)`` for each dimension except the second
      (the first and the remaining ones).
    - ``second_upper``: upper bound for the second variable (user input).

    The first variable (``epsilon``) is sampled uniformly from
    ``bounds[0]``. The second variable is then sampled uniformly from
    ``[epsilon/4, second_upper]``. All remaining variables are sampled
    independently and uniformly from their corresponding intervals
    in ``bounds[1:]``.
    """

    def __init__(self, bounds, second_upper):
        self.bounds = np.array(bounds, dtype=float)
        self.second_upper = second_upper
        self.dim = len(bounds) + 1

    def rvs(self, size=None, random_state=None):
        """Draw samples from the prior.

        - If ``size`` is None: returns a single draw.
          For this multivariate prior, that is a 1D array of shape ``(dim,)``.
        - If ``size`` is an int: returns an array of shape ``(size, dim)``.
        """
        rng = np.random.default_rng(random_state)

        if size is None:
            # Single draw, return 1D array (dim,)
            epsilon = rng.uniform(low=self.bounds[0, 0], high=self.bounds[0, 1])
            second = rng.uniform(low=epsilon / 4.0, high=self.second_upper)
            other_cols = [rng.uniform(low=a, high=b) for (a, b) in self.bounds[1:]]
            return np.asarray([epsilon, second, *other_cols], dtype=float)

        # Vectorised draws for size >= 1
        epsilon = rng.uniform(low=self.bounds[0, 0], high=self.bounds[0, 1], size=size)
        second_low = epsilon / 4.0
        second_high = np.full(size, self.second_upper, dtype=float)
        second = rng.uniform(low=second_low, high=second_high)
        other_cols = [rng.uniform(low=a, high=b, size=size) for (a, b) in self.bounds[1:]]
        if other_cols:
            others = np.column_stack(other_cols)
        else:
            # No additional dims beyond first two
            others = np.empty((size, 0), dtype=float)
        samples = np.column_stack([epsilon, second, others]).astype(float)
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

    def rvs(self, size=None, random_state=None):
        """Draw samples from an independent box-uniform prior.

        - If ``size`` is None:
          - For 1D priors (``dim == 1``), returns a Python ``float``.
          - For ``dim > 1``, returns a 1D ``ndarray`` of shape ``(dim,)``.
        - If ``size`` is an int: returns an array of shape ``(size, dim)``.
        """
        rng = np.random.default_rng(random_state)
        if size is None:
            sample = rng.uniform(self.lows, self.highs)
            sample = np.asarray(sample, dtype=float)
            if self.dim == 1:
                return float(sample.reshape(-1)[0])
            return sample
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
