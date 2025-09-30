import numpy as np
from scipy import signal
from matplotlib import pyplot as plt
import distances
import simulators
import SMCABC

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

def main():
    n = 500
    d = 0.08
    T = n*d
    # observation data comes from this:
    # data = simulators.FHN_model(initial_value = np.zeros(2), theta = [0.1, 1.5, 0.8, 0.3], timestep=0.0001, number_of_samples = 2000000)
    FHNdata = np.loadtxt("observation.txt")[0:int(T/0.0001):int(d/0.0001)]
    OHdata = simulators.OH_model(0.0, [0.0, 1, 1], d, n)

    data = OHdata
    plt.plot(data)
    plt.show()
    
    FHNprior = MultivariateUniform([(0.01,0.5),(0.01,6),(0.01,1)],6)
    OHprior = UniformND([(0.1, 10), (0.1, 10)])
   
    model_dist_calc = distances.CalculateModelBasedDistance(data, 0.08)
    pen_dist_calc = distances.CalculatePENDistance()
    # pen_dist_calc.create_and_train_PEN(
    #     model_simulator = simulators.FHN_model,
    #     training_thetas = FHNprior.rvs(10000),
    #     traj_initial_value = np.zeros(2),
    #     real_trajectory=data,
    #     timestep = d,
    #     num_epochs = 15,
    #     device_name = "cuda"
    # )

    samples, weights = SMCABC.sample_posterior( 
        threshold_percentile=0.5,
        prior=OHprior,
        data = data, timestep=0.08, distance_calculator = model_dist_calc, num_samples=1000, simulation_budget=100000, initial_value=0.0,
        model_simulator = simulators.OH_model
        )
    print(samples)
    print(np.mean(samples, axis=0))



    # Overlay prior vs posterior per dimension with scroll navigation
    prior_for_plot = OHprior
    prior_samples = prior_for_plot.rvs(size=20000)

    dims = samples.shape[1]
    fig, ax = plt.subplots()
    current_dim = [0]

    def plot_dim(d):
        ax.clear()
        # Shared range for fair overlay
        x_min = float(min(np.min(samples[:, d]), np.min(prior_samples[:, d])))
        x_max = float(max(np.max(samples[:, d]), np.max(prior_samples[:, d])))
        pad = 0.05 * (x_max - x_min + 1e-12)
        x_min -= pad
        x_max += pad
        bins = 100
        ax.hist(prior_samples[:, d], bins=bins, range=(x_min, x_max), density=True,
                histtype='step', linewidth=2, label='Prior')
        ax.hist(samples[:, d], bins=bins, range=(x_min, x_max), weights=weights,
                density=True, alpha=0.4, label='Posterior')
        ax.set_title(f'Dimension {d} - use mouse wheel to scroll')
        ax.set_xlabel('Value')
        ax.set_ylabel('Density')
        ax.legend()
        fig.canvas.draw_idle()

    def on_scroll(event):
        if event.button == 'up':
            current_dim[0] = (current_dim[0] + 1) % dims
        elif event.button == 'down':
            current_dim[0] = (current_dim[0] - 1) % dims
        plot_dim(current_dim[0])

    fig.canvas.mpl_connect('scroll_event', on_scroll)
    plot_dim(current_dim[0])
    plt.show()

if __name__ == "__main__":
    main()
