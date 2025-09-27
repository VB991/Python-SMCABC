"""Evaluate how informative the PEN-based summary statistic is for the FHN model.

Usage:
    python evaluate_pen_summary.py

The script will:
  * generate synthetic training data from the prior,
  * train the PEN summariser using `distances.CalculatePENDistance`, and
  * measure how the resulting distance correlates with parameter error.

Adjust the constants in `main()` to control dataset size, epochs, etc.
"""

import numpy as np
from scipy import stats

import simulators
import distances_PEN
from main import MultivariateUniform


def _simulate(theta, initial_value, timestep, number_of_samples):
    return simulators.FHN_model(
        initial_value=initial_value,
        theta=theta,
        timestep=timestep,
        number_of_samples=number_of_samples,
    )


def _generate_training_data(prior, num_samples, initial_value, timestep, number_of_samples, rng):
    draw_seed = int(rng.integers(0, np.iinfo(np.int32).max))
    theta_samples = prior.rvs(size=num_samples, random_state=draw_seed)
    theta_samples = np.atleast_2d(theta_samples)

    trajectories = np.empty((theta_samples.shape[0], number_of_samples), dtype=np.float32)
    for idx, theta in enumerate(theta_samples):
        traj = _simulate(theta, initial_value, timestep, number_of_samples)
        trajectories[idx] = traj.astype(np.float32)
    return trajectories, theta_samples.astype(np.float32)


def evaluate_pen_summary(
    true_theta=None,
    timestep=0.08,
    number_of_samples=625,
    initial_value=None,
    num_training_samples=10000,
    num_training_epochs=15,
    markov_order=1,
    training_batch_size=64,
    num_eval_draws=150,
    num_distance_replicates=5,
    num_true_replicates=20,
    seed=0,
    device="cuda",
):
    rng = np.random.default_rng(seed)
    if true_theta is None:
        true_theta = np.array([0.1, 1.5, 0.8, 0.3], dtype=np.float32)
    else:
        true_theta = np.asarray(true_theta, dtype=np.float32)

    if initial_value is None:
        initial_value = np.zeros(2, dtype=np.float32)
    else:
        initial_value = np.asarray(initial_value, dtype=np.float32)

    reference_traj = _simulate(true_theta, initial_value, timestep, number_of_samples)

    prior = MultivariateUniform([(0.01, 0.5), (0.01, 6.0), (0.01, 1.0)], second_upper=6.0)

    training_rng = np.random.default_rng(seed + 1)
    training_x, training_theta = _generate_training_data(
        prior=prior,
        num_samples=num_training_samples,
        initial_value=initial_value,
        timestep=timestep,
        number_of_samples=number_of_samples,
        rng=training_rng,
    )

    pen_distance = distances_PEN.CalculatePENDistance(
        real_trajectory=reference_traj,
        timestep=timestep,
        parameter_dim=true_theta.shape[0],
        training_data_x=training_x,
        training_data_params=training_theta,
        markov_order=markov_order,
        batch_size=training_batch_size,
        num_epochs=num_training_epochs,
        device=device,
    )

    eval_seed = int(rng.integers(0, np.iinfo(np.int32).max))
    eval_thetas = prior.rvs(size=num_eval_draws, random_state=eval_seed)
    eval_thetas = np.atleast_2d(eval_thetas).astype(np.float32)

    averaged_distances = np.empty(num_eval_draws, dtype=np.float32)
    parameter_errors = np.empty(num_eval_draws, dtype=np.float32)

    for idx, theta in enumerate(eval_thetas):
        replicate_distances = []
        for _ in range(num_distance_replicates):
            traj = _simulate(theta, initial_value, timestep, number_of_samples)
            replicate_distances.append(pen_distance.eval(traj))
        averaged_distances[idx] = float(np.mean(replicate_distances))
        parameter_errors[idx] = float(np.linalg.norm(theta - true_theta))

    pearson_r, pearson_p = stats.pearsonr(parameter_errors, averaged_distances)
    spearman_r, spearman_p = stats.spearmanr(parameter_errors, averaged_distances)

    same_parameter_distances = np.empty(num_true_replicates, dtype=np.float32)
    for j in range(num_true_replicates):
        traj = _simulate(true_theta, initial_value, timestep, number_of_samples)
        same_parameter_distances[j] = pen_distance.eval(traj)

    median_error = float(np.median(parameter_errors))
    close_mask = parameter_errors <= median_error
    far_mask = ~close_mask

    results = {
        "pearson_r": pearson_r,
        "pearson_p_value": pearson_p,
        "spearman_r": spearman_r,
        "spearman_p_value": spearman_p,
        "median_parameter_error": median_error,
        "median_distance_close": float(np.median(averaged_distances[close_mask])),
        "median_distance_far": float(np.median(averaged_distances[far_mask])),
        "true_distance_quantiles": np.quantile(same_parameter_distances, [0.1, 0.5, 0.9]).tolist(),
        "num_eval_draws": num_eval_draws,
        "num_training_samples": num_training_samples,
    }
    return results


def main():
    results = evaluate_pen_summary(seed=np.random.default_rng().integers(0, np.iinfo(np.int32).max))
    print("PEN summary informativeness diagnostics:\n")
    print(
        "Pearson correlation (param error vs distance): "
        f"{results['pearson_r']:.3f} (p={results['pearson_p_value']:.1e})"
    )
    print(
        "Spearman correlation: "
        f"{results['spearman_r']:.3f} (p={results['spearman_p_value']:.1e})"
    )
    print(
        "Median distance | small parameter error: "
        f"{results['median_distance_close']:.4f}"
    )
    print(
        "Median distance | large parameter error: "
        f"{results['median_distance_far']:.4f}"
    )
    q10, q50, q90 = results["true_distance_quantiles"]
    print(
        "Distance distribution for repeated sims at true parameters (10/50/90%): "
        f"{q10:.4f}, {q50:.4f}, {q90:.4f}"
    )


if __name__ == "__main__":
    main()