"""Utility to assess how informative the current summary statistic is.

Run with `python evaluate_summary_statistic.py`.
"""

import numpy as np
from scipy import stats

import simulators
import distances
from main import MultivariateUniform


def _simulate(theta, initial_value, timestep, number_of_samples):
    """Helper so we only have one place to call the simulator."""
    return simulators.FHN_model(
        initial_value=initial_value,
        theta=theta,
        timestep=timestep,
        number_of_samples=number_of_samples,
    )


def evaluate_summary(
    true_theta=None,
    timestep=0.08,
    number_of_samples=625,
    initial_value=None,
    num_prior_draws=200,
    num_replicates=20,
    seed=0,
):
    """Quantify how well distances correlate with parameter error.

    We sample parameter vectors from the prior, measure the distance between
    simulated trajectories and the reference data, and check how that distance
    relates to the Euclidean parameter error. High correlation indicates an
    informative summary statistic.
    """

    rng = np.random.default_rng(seed)
    if true_theta is None:
        true_theta = np.array([0.1, 1.5, 0.8, 0.3], dtype=float)
    else:
        true_theta = np.asarray(true_theta, dtype=float)

    if initial_value is None:
        initial_value = np.zeros(2, dtype=float)

    reference_traj = _simulate(true_theta, initial_value, timestep, number_of_samples)
    distance_calculator = distances.CalculateModelBasedDistance(
        real_trajectory=reference_traj,
        timestep=timestep,
    )

    prior = MultivariateUniform([(0.01, 0.5), (0.01, 6.0), (0.01, 1.0)], second_upper=6.0)

    prior_draw_seed = int(rng.integers(0, np.iinfo(np.int32).max))
    prior_draws = prior.rvs(size=num_prior_draws, random_state=prior_draw_seed)
    prior_draws = np.atleast_2d(prior_draws)

    distances_to_reference = np.empty(num_prior_draws, dtype=float)
    parameter_errors = np.empty(num_prior_draws, dtype=float)

    for idx, theta in enumerate(prior_draws):
        trajectory = _simulate(theta, initial_value, timestep, number_of_samples)
        distances_to_reference[idx] = distance_calculator.eval(trajectory)
        parameter_errors[idx] = np.linalg.norm(theta - true_theta)

    pearson_r, pearson_p = stats.pearsonr(parameter_errors, distances_to_reference)
    spearman_r, spearman_p = stats.spearmanr(parameter_errors, distances_to_reference)

    same_parameter_distances = np.empty(num_replicates, dtype=float)
    for j in range(num_replicates):
        trajectory = _simulate(true_theta, initial_value, timestep, number_of_samples)
        same_parameter_distances[j] = distance_calculator.eval(trajectory)

    median_error = np.median(parameter_errors)
    close_mask = parameter_errors <= median_error
    far_mask = ~close_mask

    results = {
        "pearson_r": pearson_r,
        "pearson_p_value": pearson_p,
        "spearman_r": spearman_r,
        "spearman_p_value": spearman_p,
        "num_draws": num_prior_draws,
        "median_distance_close": float(np.median(distances_to_reference[close_mask])),
        "median_distance_far": float(np.median(distances_to_reference[far_mask])),
        "median_distance_true_replicates": float(np.median(same_parameter_distances)),
        "true_replicate_quantiles": np.quantile(same_parameter_distances, [0.1, 0.5, 0.9]).tolist(),
    }
    return results


def main():
    results = evaluate_summary(seed=np.random.default_rng().integers(0, np.iinfo(np.int32).max))
    print("Summary statistic informativeness diagnostics:\n")
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
    q10, q50, q90 = results["true_replicate_quantiles"]
    print(
        "Distance distribution for repeated sims at true parameters (10/50/90%): "
        f"{q10:.4f}, {q50:.4f}, {q90:.4f}"
    )


if __name__ == "__main__":
    main()
