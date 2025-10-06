import numpy as np
from scipy import signal
from scipy.stats import gaussian_kde
from matplotlib import pyplot as plt
import OU_MCMC
import distances
import simulators
import SMCABC

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

def main(model, summary):
    n = 2500
    d = 0.08

    if model == "FHN":
        true_theta = np.array([0.1, 1.5, 0.8, 0.3])
        data = np.loadtxt("Observations\\FHN\\Observation1\\data_Delta0.08.txt")
        simulator = simulators.FHN_model
        prior = FHNMultivariateUniform([(0.01,0.5),(0.01,6),(0.01,1)],6)
        dimension_weights = 1/np.array([0.5,6,6,1])
        mkv_order = 1
        x0 = np.zeros(2)
    elif model == "OU":
        true_theta = np.array([3, 1, 1], dtype=float)
        data = simulators.OU_model(0.0, true_theta, d, n)
        simulator = simulators.OU_model
        prior = UniformND([(-5,5), (0.1, 10), (0.1, 10)])
        dimension_weights = 1/np.array([10,10,10])
        mkv_order = 1
        x0 = 0.0
    else:
        raise ValueError("Invalid model type specified")

    plt.plot(data)
    plt.show()
    
    if summary == "MODEL":
        dist_calc = distances.CalculateModelBasedDistance(data, 0.08)
    elif summary == "PEN":
        # dist_calc = distances.CalculatePENDistance()
        # dist_calc.create_and_train_PEN(
        #     markov_order=mkv_order,
        #     model_simulator = simulator,
        #     training_thetas = prior.rvs(100000),
        #     dimension_weights = dimension_weights,
        #     traj_initial_value = np.zeros(2),
        #     real_trajectory=data,
        #     timestep = d,
        #     num_epochs = 500,
        #     device_name = "cuda",
        #     early_stopping_patience=15,
        #     early_stopping_loss_drop=0.001
        # )
        # dist_calc.save_pen("TrainedNNs\\0.08,2500_FHN_PEN,weightedMSE.npz")
        dist_calc = distances.CalculatePENDistance.load_pen("TrainedNNs\\0.08,2500_FHN_PEN,weightedMSE.npz", data, d)
    else:
        raise ValueError("Invalid summary type specified")

    samples, weights = SMCABC.sample_posterior( 
        threshold_percentile=0.5,
        prior=prior,
        data = data, timestep=d, distance_calculator = dist_calc, num_samples=1000, simulation_budget=1000000,
          initial_value=x0,
        model_simulator = simulator
        )
    print("True mean:",[0.1, 1.5, 0.8, 0.3])
    print("Posterior:",np.average(samples, axis=0, weights=weights))



    # ----- Display Results ------

    # Optionally run OU-MCMC and collect a chain for overlay (OU model only)
    mcmc_chain = None
    if model == "OU":
        # Use a prior draw as initial theta to avoid biasing with true_theta
        theta_init = prior.rvs(size=1)[0]
        # Keep iterations <= 10000 to avoid adaptive branch issues in the script
        mcmc_chain = OU_MCMC.OU_MCMC(theta_init, prior, d, data, num_iterations=100000)

    dims = samples.shape[1]
    # Precompute KDE curves for all dimensions to make navigation instant
    xs_list = []
    post_y_list = []
    mcmc_y_list = []
    for d in range(dims):
        vals = samples[:, d]
        # Ensure the entire prior width is included
        if isinstance(prior, UniformND):
            vmin = float(prior.lows[d])
            vmax = float(prior.highs[d])
        elif isinstance(prior, FHNMultivariateUniform):
            if d == 0:
                vmin = float(prior.bounds[0, 0])
                vmax = float(prior.bounds[0, 1])
            elif d == 1:
                vmin = float(prior.bounds[0, 0]) / 4.0
                vmax = float(prior.second_upper)
            else:
                vmin = float(prior.bounds[d - 1, 0])
                vmax = float(prior.bounds[d - 1, 1])
        else:
            vmin = float(np.min(vals))
            vmax = float(np.max(vals))
        if mcmc_chain is not None and mcmc_chain.shape[0] > 1:
            burn = max(1000, mcmc_chain.shape[0] // 5)
            mvals = np.asarray(mcmc_chain[burn:, d]).squeeze()
            if mvals.size > 0 and np.isfinite(mvals).any():
                vmin = float(min(vmin, np.min(mvals)))
                vmax = float(max(vmax, np.max(mvals)))
        # Also include SMC samples span
        vmin = float(min(vmin, np.min(vals)))
        vmax = float(max(vmax, np.max(vals)))
        pad = 0.05 * (vmax - vmin + 1e-12)
        xmin = vmin - pad
        xmax = vmax + pad
        xs = np.linspace(xmin, xmax, 400)
        # Posterior KDE (weighted)
        kde_post = gaussian_kde(vals, weights=weights)
        post_y = kde_post(xs)
        xs_list.append(xs)
        post_y_list.append(post_y)
        # Optional MCMC KDE
        if mcmc_chain is not None and mcmc_chain.shape[0] > 1:
            try:
                kde_mcmc = gaussian_kde(mvals)
                mcmc_y = kde_mcmc(xs)
            except Exception:
                mcmc_y = None
        else:
            mcmc_y = None
        mcmc_y_list.append(mcmc_y)

    fig, ax = plt.subplots()
    current_dim = [0]

    def plot_dim(d):
        ax.clear()
        ax.plot(xs_list[d], post_y_list[d], color='C1', lw=2, label='SMCABC')
        if mcmc_y_list[d] is not None:
            ax.plot(xs_list[d], mcmc_y_list[d], color='C2', lw=2, linestyle='--', label='MCMC')
        ax.set_title(f'Dimension {d+1}')
        ax.set_xlabel('Value')
        ax.set_ylabel('Density')
        ax.axvline(true_theta[d], color='k', linestyle='--', linewidth=2, label='True value')
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
    main("FHN", "MODEL")
