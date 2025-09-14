import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import bisect
import tqdm # for progress bars
import concurrent.futures

import simulators
import distances

def sample_posterior(
           data: np.array,
           X0: float,
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



    N = 30 # number of particles kept at each iteration
    round = 0  # index for ABC rounds
    Nsim = 0 # number of simulations of SDE model
    stopping_threshold = 100
    distance_calculator = distance_calculator_class(data, timestep)
    delta = 0 # the distance tolerance level for accepting particles
    particles = np.zeros((N,4))  # array of particles
    weights = np.zeros(N)  # weights for particles
    distances_list = []  # unordered list of particle distances



    # pilot study for initial_threshold
    random_particles = prior.rvs(N)
    def simulate_and_compute_distance(particle):
        trajectory_simulation = model_simulator(X0, particle, timestep, len(data)-1)
        distance = distance_calculator.compare_trajectory(trajectory_simulation)
        return distance
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(simulate_and_compute_distance, particle)
            for particle in random_particles
        ]
        for future in tqdm.tqdm(concurrent.futures.as_completed(futures),total=N,desc="Pilot study for initial threshold"):
            distance = future.result()
            distances_list.append(distance)
    delta = np.percentile(distances_list, threshold_percentile*100)



    # initial ABC round
    round += 1
    distances_list = []  # reset distance list

    def generate_particle(i):
        distance = np.inf # distance of simulation from data
        localNsim = 0
        while distance > delta:
            trial_parameter = prior.rvs()[0]   # sample from prior
            trajectory_simulation = model_simulator(X0, trial_parameter, timestep, len(data)-1)
            distance = distance_calculator.compare_trajectory(trajectory_simulation)
            localNsim += 1
        return trial_parameter, localNsim, distance, i
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(generate_particle,i)
            for i in range(N)
        ]
        for future in tqdm.tqdm(concurrent.futures.as_completed(futures),total=N,desc="ABC round 1"):
            parameter, localNsim, distance, i = future.result()
            Nsim += localNsim
            distances_list.append(distance)
            particles[i] = parameter

    weights = np.full(N,1/N)



    # further ABC rounds
    while  Nsim < stopping_threshold:
        round += 1
        delta = np.percentile(distances_list, threshold_percentile*100)
        new_particles = np.zeros((N,4))
        new_weights = np.zeros(N)
        distances_list = []
        old_particle_selector = stats.rv_discrete(values = (np.arange(0,30), weights))  # select index for random particle
        distance = np.inf

        # calculate covariance matrix for MVN perturbation kernel; empirical covariance
        covar = particles - weights@particles  # subtract weighted mean
        covar = covar.T @ (covar*weights[:,None])
        covar = covar / (1 - np.sum(weights**2))
        covar = 0.5 * (covar + covar.T)
        sigma = 2 * covar  # covariance matrix

        def generate_perturbed_particle(i):
            distance = np.inf
            localNsim = 0
            while distance > delta:
                theta = particles[old_particle_selector.rvs()]
                proposal_sampler = stats.multivariate_normal(mean=theta, cov=sigma)
                new_theta = proposal_sampler.rvs()
                if prior.pdf(new_theta) == 0:
                    continue
                trajectory_simulation = model_simulator(X0, new_theta, timestep, len(data)-1)
                localNsim += 1
                distance = distance_calculator.compare_trajectory(trajectory_simulation)
            return new_theta, localNsim, distance, i

        # generate N new particles
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(generate_perturbed_particle, i)
                for i in range(N)
            ]
            for future in tqdm.tqdm(concurrent.futures.as_completed(futures),total=N,desc="ABC round {}".format(round)):
                new_particle, localNsim, distance, i = future.result()
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
