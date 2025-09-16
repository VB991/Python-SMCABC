import numpy as np
from scipy import stats
from scipy.special import logsumexp
import tqdm  # for progress bars
import concurrent.futures

import simulators
import distances

# ---- Globals (fixed once for all workers) ----
X0 = None
TIMESTEP = None
DATA_LENGTH = None
MODEL_SIMULATOR = None
DISTANCE_CALCULATOR = None
PRIOR = None


def init_globals(x0, timestep, data_length, model_simulator, distance_calculator, prior):
    """Initialise global objects once per worker process."""
    global X0, TIMESTEP, DATA_LENGTH, MODEL_SIMULATOR, DISTANCE_CALCULATOR, PRIOR
    X0 = x0
    TIMESTEP = timestep
    DATA_LENGTH = data_length
    MODEL_SIMULATOR = model_simulator
    DISTANCE_CALCULATOR = distance_calculator
    PRIOR = prior


# ---- Worker helpers ----

def pilot_study_worker(indexes):
    random_particles = PRIOR.rvs(len(indexes))
    results = []
    for particle in random_particles:
        traj = MODEL_SIMULATOR(X0, particle, TIMESTEP, DATA_LENGTH)
        dist = DISTANCE_CALCULATOR.compare_trajectory(traj)
        results.append(dist)
    return results


def initial_ABC_worker(indexes, distance_threshold):
    results = []
    for i in indexes:
        dist = np.inf
        local_nsim = 0
        while dist > distance_threshold:
            trial_parameter = PRIOR.rvs()[0]
            traj = MODEL_SIMULATOR(X0, trial_parameter, TIMESTEP, DATA_LENGTH)
            dist = DISTANCE_CALCULATOR.compare_trajectory(traj)
            local_nsim += 1
        results.append((trial_parameter, local_nsim, dist, i))
    return results


def SMCABC_worker(indexes, distance_threshold, kernel_covariance, random_particle_selector, particles):
    dim = particles.shape[1]
    proposal_kernel = stats.multivariate_normal(mean=np.zeros(dim), cov=kernel_covariance)
    results = []
    for i in indexes:
        distance = np.inf
        local_nsim = 0
        while distance > distance_threshold:
            ancestor = particles[random_particle_selector.rvs()]
            new_theta = ancestor + proposal_kernel.rvs()
            if PRIOR.pdf(new_theta) == 0:
                continue
            trajectory_simulation = MODEL_SIMULATOR(X0, new_theta, TIMESTEP, DATA_LENGTH)
            local_nsim += 1
            distance = DISTANCE_CALCULATOR.compare_trajectory(trajectory_simulation)
        results.append((new_theta, local_nsim, distance, i))
    return results


# ---- Main routine ----

def sample_posterior(
    data: np.array,
    initial_value: float,
    timestep: float,
    threshold_percentile: float,
    prior: stats.rv_continuous,
    model_simulator: callable = simulators.FHN_model,
    distance_calculator_class: type = callable,
):
    """Performs Sequential Monte Carlo ABC."""

    N = 1000  # number of particles kept at each iteration
    Nsim = 0  # number of simulations of SDE model
    stopping_threshold = 10000  # maximum number of simulations
    batch_size = 1

    round_idx = 0  # index for ABC rounds
    distance_calculator = distance_calculator_class(data, timestep)
    distance_threshold = None  # distance tolerance
    particles = np.zeros((N, 4))
    weights = np.zeros(N)
    distances_list = []

    with concurrent.futures.ProcessPoolExecutor(
        initializer=init_globals,
        initargs=(initial_value, timestep, len(data), model_simulator, distance_calculator, prior),
    ) as executor:
        # Pilot study
        batches = [list(range(i, min(i + batch_size, N))) for i in range(0, N, batch_size)]
        futures = [executor.submit(pilot_study_worker, b) for b in batches]
        for future in tqdm.tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Pilot study"):
            distances = future.result()
            distances_list.extend(distances)
        distance_threshold = np.percentile(distances_list, threshold_percentile * 100)

        # Initial ABC round
        round_idx += 1
        distances_list = []
        futures = [executor.submit(initial_ABC_worker, b, distance_threshold) for b in batches]
        for future in tqdm.tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="ABC round 1"):
            for parameter, local_nsim, distance, idx in future.result():
                Nsim += local_nsim
                distances_list.append(distance)
                particles[idx] = parameter
        weights = np.full(N, 1 / N)

        # Subsequent rounds
        while Nsim < stopping_threshold:
            round_idx += 1
            distance_threshold = np.percentile(distances_list, threshold_percentile * 100)
            new_particles = np.zeros((N, 4))
            distances_list = []

            # Build zero-mean kernel for this round
            centred = particles - weights @ particles
            cov = centred.T @ (centred * weights[:, None])
            cov /= (1 - np.sum(weights ** 2))
            cov = 0.5 * (cov + cov.T)
            sigma = 2 * cov
            zero_mean_kernel = stats.multivariate_normal(mean=np.zeros(particles.shape[1]), cov=sigma)

            ancestor_selector = stats.rv_discrete(values=(np.arange(0, N), weights))
            futures = [
                executor.submit(
                    SMCABC_worker,
                    batch,
                    distance_threshold,
                    sigma,
                    ancestor_selector,
                    particles,
                )
                for batch in batches
            ]

            for future in tqdm.tqdm(
                concurrent.futures.as_completed(futures), total=len(futures), desc=f"ABC round {round_idx}"
            ):
                for new_particle, local_nsim, distance, idx in future.result():
                    Nsim += local_nsim
                    distances_list.append(distance)
                    new_particles[idx] = new_particle

            # Calculate new weights
            # Using log-sum-exp trick for numerical stability
            diffs = new_particles[:, None, :] - particles[None, :, :]
            flat_diffs = diffs.reshape(-1, particles.shape[1])
            log_kernel = zero_mean_kernel.logpdf(flat_diffs).reshape(N, N)
            log_denominator = logsumexp(log_kernel, b=weights, axis=1)
            denominators = np.exp(log_denominator)
            numerators = prior.pdf(new_particles)
            new_weights = numerators / np.maximum(denominators, 1e-300)

            particles = new_particles
            weights = new_weights / np.sum(new_weights)

    return particles, weights
