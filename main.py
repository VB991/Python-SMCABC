import json
from pathlib import Path
import numpy as np
from scipy import signal
from scipy.stats import gaussian_kde
import matplotlib as mpl
mpl.use("Agg")
from matplotlib import pyplot as plt
import OU_MCMC
import distances
import simulators
import SMCABC
from priors import MultivariateUniform, UniformND

def _save_results(output_path, samples, weights, meta):
    """Save SMCABC results to the given path.

    Supports `.npz` (compressed) and `.csv`. For `.csv`, also writes a
    sidecar `.json` with metadata.
    """
    if not output_path:
        return
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    try:
        if p.suffix.lower() == ".csv":
            arr = np.hstack([samples, weights.reshape(-1, 1)])
            header_cols = [f"theta_{i}" for i in range(samples.shape[1])] + ["weight"]
            header = ",".join(header_cols)
            np.savetxt(p, arr, delimiter=",", header=header, comments="")
            # Write metadata sidecar
            meta_path = p.with_suffix(".json")
            with meta_path.open("w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)
        else:
            # Default to NPZ for anything else (.npz recommended)
            np.savez_compressed(
                p,
                samples=samples,
                weights=weights,
                meta=json.dumps(meta),
            )
    except Exception as e:
        print(f"Warning: failed to save results to '{p}': {e}")


def main(model, summary, output_path=None):
    n = 2500
    d = 0.08
    T = n*d
    # observation data comes from this:
    # data = simulators.FHN_model(initial_value = np.zeros(2), theta = [0.1, 1.5, 0.8, 0.3], timestep=0.0001, number_of_samples = 2000000)
    # np.savetxt("observation.txt", data)

    if model == "FHN":
        true_theta = np.array([0.1, 1.5, 0.8, 0.3])
        data = np.loadtxt("data_Delta0.08.txt")[0:n]
        simulator = simulators.FHN_model
        prior = MultivariateUniform([(0.01,0.5),(0.01,6),(0.01,1)],6)
        mkv_order = 1
        x0 = np.zeros(2)
    elif model == "OU":
        true_theta = np.array([3, 1, 1], dtype=float)
        data = simulators.OU_model(0.0, true_theta, d, n)
        simulator = simulators.OU_model
        prior = UniformND([(-5,5), (0.1, 10), (0.1, 10)])
        mkv_order = 0
        x0 = 0.0
    else:
        raise ValueError("Invalid model type specified")

    # Avoid interactive plotting before multiprocessing (can clash with TkAgg)
    
    if summary == "MODEL":
        dist_calc = distances.CalculateModelBasedDistance(data, 0.08)
    elif summary == "PEN":
        dist_calc = distances.CalculatePENDistance()
        dist_calc.create_and_train_PEN(
            markov_order=mkv_order,
            model_simulator = simulator,
            training_thetas = prior.rvs(10000),
            traj_initial_value = np.zeros(2),
            real_trajectory=data,
            timestep = d,
            num_epochs = 50,
            device_name = "cuda",
            early_stopping_patience=5,
            early_stopping_loss_drop=0.1
        )
    else:
        raise ValueError("Invalid summary type specified")

    # Ensure no open figures before starting multiprocessing
    plt.close('all')
    samples, weights = SMCABC.sample_posterior( 
        threshold_percentile=0.5,
        prior=prior,
        data = data, timestep=d, distance_calculator = dist_calc, num_samples=1000, simulation_budget=1000000,
          initial_value=x0,
        model_simulator = simulator
        )
    # Persist results if requested
    meta = {
        "model": model,
        "summary": summary,
        "timestep": float(d),
        "n_samples": int(samples.shape[0]),
        "dims": int(samples.shape[1]),
        "true_theta": [float(x) for x in np.asarray(true_theta).tolist()],
    }
    _save_results(output_path, samples, weights, meta)

    print("True mean:", true_theta.tolist())
    print("Posterior:", np.average(samples, axis=0, weights=weights))



    # ----- Display Results ------

    # Optionally run OU-MCMC and collect a chain for overlay (OU model only)
    mcmc_chain = None
    if model == "OU":
        # Use a prior draw as initial theta to avoid biasing with true_theta
        theta_init = prior.rvs(size=1)[0]
        # Keep iterations <= 10000 to avoid adaptive branch issues in the script
        mcmc_chain = OU_MCMC.OU_MCMC(theta_init, prior, d, data, num_iterations=1000000)

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
        elif isinstance(prior, MultivariateUniform):
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
            ax.plot(xs_list[d], mcmc_y_list[d], color='C2', lw=2, linestyle='--', label='OU-MCMC (KDE)')
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
