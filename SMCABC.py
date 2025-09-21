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
        dist = DISTANCE_CALCULATOR.eval(traj)
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
            dist = DISTANCE_CALCULATOR.eval(traj)
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
            distance = DISTANCE_CALCULATOR.eval(trajectory_simulation)
        results.append((new_theta, local_nsim, distance, i))
    return results


# ---- Main routine ----

def sample_posterior(
    data: np.array,
    timestep: float,
    threshold_percentile: float,
    prior: stats.rv_continuous,
    model_simulator: callable = simulators.FHN_model,
    distance_calculator: type = callable,
    num_samples: int = 100,
    simulation_budget: int = 10000,
):
    """_summary_

    Args:
        data (np.array): Real data time-series
        timestep (float): Time distance between consecutive samples in data
        threshold_percentile (float): Percentile (0,1) for updating distance threshold
        prior (stats.rv_continuous): Prior distribution for parameter
        model_simulator (callable, optional): Simulator for model expected to generate data from simulators module. Defaults to simulators.FHN_model.
        distance_calculator (type, optional): Distance calculator instance from distance module. Defaults to callable.
        num_samples (int, optional): Number of particles to use. Defaults to 1000.
        simulation_budget (int, optional): Maximum number of model simulations to perform. Defaults to 10000.
    Returns:
        _type_: _description_
    """

    N = num_samples  # number of particles kept at each iteration
    batch_size = 1
    Nsim = 0  # number of simulations of SDE model


    particle_dimension = len(np.squeeze(np.atleast_1d(prior.rvs()))) # get the shape of the particles
    round_idx = 0  # index for ABC rounds
    initial_value = data[0]
    distance_threshold = None  # distance tolerance
    particles = np.empty((N, particle_dimension))
    weights = np.empty(N)
    distances_list = []

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=None,
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
        while Nsim < simulation_budget:
            round_idx += 1
            distance_threshold = np.percentile(distances_list, threshold_percentile * 100)
            new_particles = np.empty((N, particle_dimension))
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

            # Update weights
            # For below: (new_particle, repeat, pdim) - (repeat, old_particle, pdim)
            diffs = np.repeat(new_particles[:,None,:], repeats=N, axis=1) - np.repeat(particles[None,:,:], repeats=N, axis=0)
            diffs = diffs.reshape(-1, particle_dimension)   # (new_particle, index, pdim) --> (new_particle*index, pdim)
            kernel_vals = zero_mean_kernel.pdf(diffs).reshape(N, N)  # (new_particle, index) array of kernel values
            denominators = np.sum(weights * kernel_vals, axis = 1)
            new_weights = prior.pdf(new_particles) / denominators

            particles = new_particles
            weights = new_weights / np.sum(new_weights)

    return particles, weights
