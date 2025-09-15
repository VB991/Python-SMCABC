import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import tqdm # for progress bars
import concurrent.futures

import simulators
import distances

# global variables to be used by worker processes
X0 = None
TIMESTEP = None
DATA_LENGTH = None
MODEL_SIMULATOR = None
DISTANCE_CALCULATOR = None
PRIOR = None
DISTANCE_THRESHOLD = None
KERNEL_COVARIANCE = None
RANDOM_PARTICLE_SELECTOR = None
PARTICLES = None

def init_globals(x0, 
                timestep,
                data_length, 
                model_simulator, 
                distance_calculator, 
                prior, 
                distance_threshold,
                kernel_covariance, 
                random_particle_selector, 
                particles):
    global X0, TIMESTEP, DATA_LENGTH, MODEL_SIMULATOR, DISTANCE_CALCULATOR, PRIOR, DISTANCE_THRESHOLD, KERNEL_COVARIANCE, RANDOM_PARTICLE_SELECTOR, PARTICLES
    X0 = x0
    TIMESTEP = timestep
    DATA_LENGTH = data_length
    MODEL_SIMULATOR = model_simulator
    DISTANCE_CALCULATOR = distance_calculator
    PRIOR = prior
    DISTANCE_THRESHOLD = distance_threshold
    KERNEL_COVARIANCE = kernel_covariance
    RANDOM_PARTICLE_SELECTOR = random_particle_selector
    PARTICLES = particles

# worker processes for SMC-ABC

def pilot_study_worker(indexes):
    random_particles = PRIOR.rvs(len(indexes))
    results = []
    for particle in random_particles:
        traj = MODEL_SIMULATOR(X0, particle, TIMESTEP, DATA_LENGTH)
        dist = DISTANCE_CALCULATOR.compare_trajectory(traj)
        results.append(dist)
    return results

def initial_ABC_worker(indexes):
    results = []
    for i in indexes:
        dist = np.inf
        localNsim = 0
        while dist > DISTANCE_THRESHOLD:
            trial_parameter = PRIOR.rvs()[0]
            traj = MODEL_SIMULATOR(X0, trial_parameter, TIMESTEP, DATA_LENGTH)
            dist = DISTANCE_CALCULATOR.compare_trajectory(traj)
            localNsim += 1
        results.append((trial_parameter, localNsim, dist, i))
    return results

def SMCABC_worker(indexes):
    results = []
    for i in indexes:
        distance = np.inf
        localNsim = 0
        while distance > DISTANCE_THRESHOLD:
            theta = PARTICLES[RANDOM_PARTICLE_SELECTOR.rvs()]
            proposal_sampler = stats.multivariate_normal(mean=theta, cov=KERNEL_COVARIANCE)
            new_theta = proposal_sampler.rvs()
            if PRIOR.pdf(new_theta) == 0:
                continue
            trajectory_simulation = MODEL_SIMULATOR(X0, new_theta, TIMESTEP, DATA_LENGTH)
            localNsim += 1
            distance = DISTANCE_CALCULATOR.compare_trajectory(trajectory_simulation)
        results.append((new_theta, localNsim, distance, i))
    return results



def sample_posterior(
           data: np.array,
           initial_value: float,
           timestep: float,
           threshold_percentile: float,
           prior: stats.rv_continuous,
           model_simulator: callable = simulators.FHN_model,
           distance_calculator_class: type = callable
           ):
    """Performs Sequential Monte Carlo ABC

    Args:
        prior (stats.rv_continuous): Prior distribution used for ABC
        data (np.array): Equally spaced samples from random process for estimation to be performed on
        delta_1 (float): Initial tolerance level for accepting particles
        model_simulator (callable, optional): Simulator for the SDE generating the data. Defaults to simulators.FHN_model.
        distance_function (callable, optional): Distance function and summary statistic to be used in ABC. Defaults to distances.model_based_summary_distance.
    """

    N = 100 # number of particles kept at each iteration
    Nsim = 0 # number of simulations of SDE model
    stopping_threshold = 1000  # maximum number of simulations
    batch_size = 10

    round = 0  # index for ABC rounds
    distance_calculator = distance_calculator_class(data, timestep)
    distance_threshold = 0 # the distance tolerance level for accepting particles
    particles = np.zeros((N,4))  # array of particles
    weights = np.zeros(N)  # weights for particles
    distances_list = []  # unordered list of particle distances

    init_globals(initial_value, timestep, len(data), model_simulator, distance_calculator, prior, None, None, None, None)

    # ----- Pilot study for initial threshold -----
    with concurrent.futures.ProcessPoolExecutor(
        initializer=init_globals,
        initargs=(initial_value, timestep, len(data), model_simulator, distance_calculator, prior, None, None, None, None)
    ) as executor:
        batches = [list(range(i, min(i + batch_size, N))) for i in range(0, N, batch_size)]
        futures = [executor.submit(pilot_study_worker, b) for b in batches]
        for future in tqdm.tqdm(concurrent.futures.as_completed(futures),total=len(futures),desc="Pilot study for initial threshold"):
            distances = future.result()
            distances_list = distances_list + distances
    distance_threshold = np.percentile(distances_list, threshold_percentile*100)

    # ----- Initial ABC round -----
    round += 1
    distances_list = []  # reset distance list
    with concurrent.futures.ProcessPoolExecutor(
        initializer=init_globals,
        initargs=(initial_value, timestep, len(data), model_simulator, distance_calculator, prior, distance_threshold, None, None, None)
    ) as executor:
        batches = [list(range(i, min(i + batch_size, N))) for i in range(0, N, batch_size)]
        futures = [executor.submit(initial_ABC_worker, b) for b in batches]
        for future in tqdm.tqdm(concurrent.futures.as_completed(futures),total=len(futures),desc="ABC round 1"):
            for result in future.result():
                parameter, localNsim, distance, i = result
                Nsim += localNsim
                distances_list.append(distance)
                particles[i] = parameter
    weights = np.full(N,1/N)

    # ------ Further ABC rounds -----
    while  Nsim < stopping_threshold:
        round += 1
        distance_threshold = np.percentile(distances_list, threshold_percentile*100)
        new_particles = np.zeros((N,4))
        new_weights = np.zeros(N)
        distances_list = []
        old_particle_selector = stats.rv_discrete(values = (np.arange(0,N), weights))  # select index for random particle
        distance = np.inf
    
        # calculate covariance matrix for MVN perturbation kernel; empirical covariance
        c = particles - weights@particles  # subtract weighted mean
        c = c.T @ (c*weights[:,None])
        c = c / (1 - np.sum(weights**2))
        c = 0.5 * (c + c.T)
        sigma = 2 * c  # covariance matrix

        with concurrent.futures.ProcessPoolExecutor(
            initializer=init_globals,
            initargs=(initial_value, timestep, len(data), model_simulator, distance_calculator, prior, distance_threshold, sigma, old_particle_selector, particles)
        ) as executor:
            batches = [list(range(i, min(i + batch_size, N))) for i in range(0, N, batch_size)]
            futures = [executor.submit(SMCABC_worker, b) for b in batches]
            for future in tqdm.tqdm(concurrent.futures.as_completed(futures),total=len(futures),desc="ABC round {}".format(round)):
                for result in future.result():
                    new_particle, localNsim, distance, i = result
                    Nsim += localNsim
                    distances_list.append(distance)

                new_particles[i] = new_particle
                temp = np.array([
                    (lambda i: weights[int(i)]*stats.multivariate_normal(mean=particles[i],cov=sigma).pdf(new_particle))(i) for i in range(N)
                    ])
                new_weights[i] = prior.pdf(new_particle) / np.sum(temp)
            
        particles = new_particles
        weights = new_weights / np.sum(new_weights)
        


    return particles, weights
