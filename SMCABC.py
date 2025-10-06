import numpy as np
from scipy.stats import multivariate_normal, rv_continuous, rv_discrete
import os, time
import tqdm  # for progress bars
import concurrent.futures
import simulators

# ---- Globals (fixed once for all workers) ----
X0 = None
TIMESTEP = None
DATA_LENGTH = None
MODEL_SIMULATOR = None
DISTANCE_CALCULATOR = None
PRIOR = None
RNG = None
OLCM_PARTICLES = None
OLCM_WEIGHTS = None


def init_globals(x0, timestep, data_length, model_simulator, distance_calculator, prior):
    """Initialise global objects once per worker process."""
    global X0, TIMESTEP, DATA_LENGTH, MODEL_SIMULATOR, DISTANCE_CALCULATOR, PRIOR, RNG
    X0 = x0
    TIMESTEP = timestep
    DATA_LENGTH = data_length
    MODEL_SIMULATOR = model_simulator
    DISTANCE_CALCULATOR = distance_calculator
    PRIOR = prior
    # Per-process RNG to avoid identical random streams across workers
    seed = (os.getpid() << 32) ^ time.time_ns()
    RNG = np.random.default_rng(seed)

def init_olcm_covariances(olcm_particles, olcm_weights):
    """Initializer for covariance-building pool to set OLCM arrays once per worker."""
    global OLCM_PARTICLES, OLCM_WEIGHTS
    OLCM_PARTICLES = np.asarray(olcm_particles, dtype=float)
    OLCM_WEIGHTS = np.asarray(olcm_weights, dtype=float)

def build_sigma_for_particle(particle):
    """Compute weighted outer-product covariance for one particle using globals."""
    particle = np.asarray(particle, dtype=float)
    diffs = OLCM_PARTICLES - particle  # (K, d)
    weighted_diffs = OLCM_WEIGHTS[:, None] * diffs
    sigma = diffs.T @ weighted_diffs
    d = particle.size
    sigma += 1e-8 * np.eye(d)
    return sigma


# ---- Worker helpers ----

def pilot_study_worker():
    particle = PRIOR.rvs(random_state=RNG)
    traj = MODEL_SIMULATOR(X0, particle, TIMESTEP, DATA_LENGTH)
    # Catch infinite trajectory
    if np.all(np.isfinite(traj)):
        dist = DISTANCE_CALCULATOR.eval(traj)
    else:
        dist = np.inf   # set huge trajectories as inf far away
    result = dist
    return result


def initial_ABC_worker(index, distance_threshold):
    dist = np.inf
    while dist > distance_threshold:
        trial_parameter = PRIOR.rvs(random_state=RNG)
        traj = MODEL_SIMULATOR(X0, trial_parameter, TIMESTEP, DATA_LENGTH)
        if np.all(np.isfinite(traj)):
            dist = DISTANCE_CALCULATOR.eval(traj)
        else:
            dist = np.inf
    result = (trial_parameter, dist, index)
    return result


def SMCABC_worker(index, distance_threshold, kernels, random_particle_selector, particles):
    dist = np.inf
    while dist > distance_threshold:
        random_particle_index = random_particle_selector.rvs(random_state=RNG)
        ancestor = particles[random_particle_index]
        # Draw directly from kernel centered at ancestor to avoid double-shifting
        new_theta = kernels[random_particle_index].rvs(random_state=RNG)
        if PRIOR.pdf(new_theta) == 0:
            continue
        trajectory_simulation = MODEL_SIMULATOR(X0, new_theta, TIMESTEP, DATA_LENGTH)
        if np.all(np.isfinite(trajectory_simulation)):
            dist = DISTANCE_CALCULATOR.eval(trajectory_simulation)
        else:
            dist = np.inf
    result = (new_theta, dist, index)
    return result


def weight_update_worker(index, new_particle, kernels, old_particles, old_weights):
    k_vals = np.empty(len(old_particles), dtype=float)
    for l in range(len(old_particles)):
        k_vals[l] = float(kernels[l].pdf(new_particle))
    k_vals = np.array(k_vals)
    denom = np.sum(old_weights * k_vals)
    result = (index, denom)
    return result


# ---- Main routine ----

def sample_posterior(
    data: np.array,
    initial_value: np.array,
    timestep: float,
    threshold_percentile: float,
    prior: rv_continuous,
    model_simulator: callable,
    distance_calculator: type = callable,
    num_samples: int = 100,
    simulation_budget: int = 10000,
):
    """_summary_

    Args:
        data (np.array): Real data time-series
        initial_value (np.array): Initial value of data (all, including non-observed components)
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
    Nsim = 0  # number of simulations of SDE model
    pbar = tqdm.tqdm(total=simulation_budget, desc="NSim", unit="sim", position=0, leave=True)

    particle_dimension = len(np.squeeze(np.atleast_1d(prior.rvs()))) # get the shape of the particles
    round_idx = 0  # index for ABC rounds
    distance_threshold = None  # distance tolerance
    particles = np.empty((N, particle_dimension))
    weights = np.empty(N)
    distances = np.empty(N)

    # Initialise worker processes
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=None,
        initializer=init_globals,
        initargs=(initial_value, timestep, len(data), model_simulator, distance_calculator, prior),
    ) as executor:
        
        # ------ Pilot study -------
        pilot_distances = []
        futures = [executor.submit(pilot_study_worker) for i in range(N)]
        for future in tqdm.tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Pilot study", position=1, leave=False):
            distance = future.result()
            pilot_distances.append(distance)
        distance_threshold = np.percentile(pilot_distances, threshold_percentile * 100)

        # ------- Initial ABC round -------
        round_idx += 1
        futures = [executor.submit(initial_ABC_worker, index, distance_threshold) for index in range(N)]
        for future in tqdm.tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc=f"ABC round 1: δ={distance_threshold:.3f}", position=1, leave=False):
            parameter, distance, idx = future.result()
            Nsim += 1
            pbar.update(1)
            distances[idx] = distance
            particles[idx] = parameter
        weights = np.full(N, 1 / N)

        # ------- Subsequent rounds -------
        new_particles = particles.copy()
        new_weights = weights.copy()
        while Nsim < simulation_budget:
            round_idx += 1
            distance_threshold = np.percentile(distances, threshold_percentile * 100)

            # Build kernel covariances for this round
            mask = distances < distance_threshold
            olcm_particles = particles[mask]
            olcm_weights  = weights[mask]
            olcm_weights /= np.sum(olcm_weights)
            # Build kernel covariances with a straightforward loop (no nested pools)
            olcm_kernels = []
            size = np.size(particles[0])
            for j in range(N):
                particle = particles[j]
                sigma = np.zeros((size, size))
                for l in range(len(olcm_particles)):
                    diff = olcm_particles[l] - particle
                    diff = diff[:, np.newaxis]
                    sigma += olcm_weights[l] * (diff @ diff.T)
                olcm_kernels.append(multivariate_normal(mean=particle, cov=sigma, allow_singular=True))

            particles = new_particles
            weights = new_weights

            ancestor_selector = rv_discrete(values=(np.arange(0, N), weights))
            futures = [
                executor.submit(
                    SMCABC_worker,
                    index,
                    distance_threshold,
                    olcm_kernels,
                    ancestor_selector,
                    particles,
                )
                for index in range(N)
            ]
            for future in tqdm.tqdm(
                concurrent.futures.as_completed(futures), total=len(futures), desc=f"ABC round {round_idx}: δ={distance_threshold:.3f} ", position=1, leave=False
            ):
                new_particle, distance, idx = future.result()
                Nsim += 1
                pbar.update(1)
                distances[idx] = distance
                new_particles[idx] = new_particle

            futures = [
                executor.submit(
                    weight_update_worker,
                    index,
                    new_particles[index],
                    olcm_kernels,
                    particles,
                    weights,
                )
                for index in range(N)
            ]

            denominators = np.empty(N, dtype=float)
            for future in concurrent.futures.as_completed(futures):
                idx, denom = future.result()
                denominators[idx] = denom

            denominators[denominators==0] = 1e-5    # Catch / by 0
            new_weights = prior.pdf(new_particles) / denominators
            new_weights /= np.sum(new_weights)

            # Ensure latest generation is used thereafter and on return
            particles = new_particles
            weights = new_weights


    return particles, weights
