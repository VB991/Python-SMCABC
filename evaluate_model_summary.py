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

    reference_traj = np.loadtxt("observation.txt")[0:int(number_of_samples*timestep/0.0001):int(timestep/0.0001)]
    distance_calculator = distances.CalculateModelBasedDistance(
        real_trajectory=reference_traj,
        timestep=timestep,
    )

    prior = MultivariateUniform([(0.01, 0.5), (0.01, 6.0), (0.01, 1.0)], second_upper=6.0)

    prior_draw_seed = int(rng.integers(0, np.iinfo(np.int32).max))
    prior_draws = np.atleast_2d(prior.rvs(size=num_prior_draws, random_state=prior_draw_seed))

    # Collect only finite simulations to avoid KDE failures
    distances_list = []
    param_error_list = []

    for theta in prior_draws:
        trajectory = _simulate(theta, initial_value, timestep, number_of_samples)
        if np.all(np.isfinite(trajectory)):
            try:
                d = distance_calculator.eval(trajectory)
            except Exception:
                # In case downstream distance calc fails for any reason, skip
                continue
            if np.isfinite(d):
                distances_list.append(float(d))
                param_error_list.append(float(np.linalg.norm(theta - true_theta)))

    distances_to_reference = np.asarray(distances_list, dtype=float)
    parameter_errors = np.asarray(param_error_list, dtype=float)

    if parameter_errors.size >= 2:
        pearson_r, pearson_p = stats.pearsonr(parameter_errors, distances_to_reference)
        spearman_r, spearman_p = stats.spearmanr(parameter_errors, distances_to_reference)
    else:
        pearson_r = spearman_r = np.nan
        pearson_p = spearman_p = np.nan

    same_parameter_distances = []
    for _ in range(num_replicates):
        trajectory = _simulate(true_theta, initial_value, timestep, number_of_samples)
        if np.all(np.isfinite(trajectory)):
            try:
                d = distance_calculator.eval(trajectory)
            except Exception:
                continue
            if np.isfinite(d):
                same_parameter_distances.append(float(d))
    same_parameter_distances = np.asarray(same_parameter_distances, dtype=float)

    if parameter_errors.size > 0:
        median_error = float(np.median(parameter_errors))
        close_mask = parameter_errors <= median_error
        far_mask = ~close_mask
    else:
        median_error = np.nan
        close_mask = np.array([], dtype=bool)
        far_mask = np.array([], dtype=bool)

    results = {
        "pearson_r": pearson_r,
        "pearson_p_value": pearson_p,
        "spearman_r": spearman_r,
        "spearman_p_value": spearman_p,
        "num_draws": num_prior_draws,
        "median_distance_close": float(np.median(distances_to_reference[close_mask])) if close_mask.size and np.any(close_mask) else np.nan,
        "median_distance_far": float(np.median(distances_to_reference[far_mask])) if far_mask.size and np.any(far_mask) else np.nan,
        "median_distance_true_replicates": float(np.median(same_parameter_distances)) if same_parameter_distances.size else np.nan,
        "true_replicate_quantiles": np.quantile(same_parameter_distances, [0.1, 0.5, 0.9]).tolist() if same_parameter_distances.size else [np.nan, np.nan, np.nan],
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
