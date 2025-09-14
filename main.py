import numpy as np
from scipy import signal
from matplotlib import pyplot as plt
import distances
import simulators
import SMCABC

data = simulators.FHN_model(X0 = np.zeros(2), theta = [0.1, 1.5, 0.8, 0.3], timestep=0.08, N = 625)
plt.plot(data)
plt.show()

class MultivariateUniform:
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

samples, weights = SMCABC.sample_posterior(
    X0=[0,0], 
    threshold_percentile=0.5,
    prior=MultivariateUniform([(0.01,0.5),(0.01,6),(0.01,1)],6),
    data = data, timestep=0.1, distance_calculator_class=distances.CalculateModelBasedDistance)
print(samples)
print(np.mean(samples, axis=0))