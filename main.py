from datetime import datetime
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

import OU_MCMC
import SMCABC
import distances
import simulators
from kde_utils import (
    build_kde_grids,
    evaluate_mcmc_kdes,
    evaluate_weighted_kdes,
    plot_saved_kde_runs,
    save_kde_results,
)
from priors import FHNMultivariateUniform, UniformND

def main(model, summary, kde_output_dir=None):
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
    grids = build_kde_grids(prior, dims)
    smcabc_kdes = evaluate_weighted_kdes(samples, weights, grids)
    mcmc_kdes = evaluate_mcmc_kdes(mcmc_chain, grids)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    metadata = {
        "model": model,
        "summary": summary,
        "timestamp": timestamp,
        "num_samples": int(samples.shape[0]),
        "num_grid_points": int(grids.shape[1]),
        "true_theta": true_theta.tolist(),
    }

    if kde_output_dir is None:
        output_dir = Path("Experimental Results") / "KDEs"
    else:
        output_dir = Path(kde_output_dir)
    filename = f"{model}_{summary}_{timestamp}.npz"
    kde_path = save_kde_results(output_dir / filename, grids, smcabc_kdes, metadata, mcmc_kdes)
    print(f"Saved KDE evaluations to {kde_path}")

    plot_saved_kde_runs([kde_path], true_theta=true_theta)

if __name__ == "__main__":
    # Update this path to route KDE outputs elsewhere if desired
    KDE_OUTPUT_DIR = Path("Experimental Results") / "KDEs"
    main("FHN", "MODEL", kde_output_dir="Experimental Results\\FHN-model\\Observation1-delta0.08")
