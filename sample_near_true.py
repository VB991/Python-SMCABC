import numpy as np

import simulators
import distances
import distances_PEN
from tqdm import tqdm
from main import MultivariateUniform


def load_data(n=625, d=0.08, path="observation.txt"):
    # Subsample as in main.py
    return np.loadtxt(path)[0:int(n * d / 0.0001):int(d / 0.0001)]


def draw_near_true(rng, true_theta):
    # Gaussian perturbation around the true parameter
    # Standard deviations chosen to stay close while exploring
    std = np.array([0.01, 0.15, 0.08, 0.03], dtype=float)
    theta = true_theta + rng.normal(scale=std, size=true_theta.shape)

    # Clip to the same support used elsewhere in the repo
    # epsilon in [0.01, 0.5]
    theta[0] = np.clip(theta[0], 0.01, 0.5)
    # gamma in [epsilon/4, 6]
    theta[1] = np.clip(theta[1], theta[0] / 4.0, 6.0)
    # beta in [0.01, 6]
    theta[2] = np.clip(theta[2], 0.01, 6.0)
    # sigma in [0.01, 1]
    theta[3] = np.clip(theta[3], 0.01, 1.0)
    return theta


def main():
    # Set as in main.py
    n = 2500
    d = 0.08
    true_theta = np.array([0.1, 1.5, 0.8, 0.3], dtype=float)

    rng = np.random.default_rng(0)

    data = load_data(n=n, d=d)

    train_data = []
    sampler = MultivariateUniform([(0.01,0.5),(0.01,6),(0.01,1)],6)
    thetas = sampler.rvs(size=1000)
    for theta in thetas:
        train_data.append(simulators.FHN_model(np.zeros(2), theta, timestep=d, number_of_samples=n))
    train_data = np.array(train_data)
    dist_calc = distances_PEN.CalculatePENDistance(real_trajectory=data, parameter_dim=4, timestep=d, training_data_x=train_data, training_data_params=thetas, device="cuda")

    kept = []
    kept_dists = []
    attempts = 0
    threshold = 0.01

    # Two progress bars: total simulations (open-ended) and accepted draws (target 50)
    sim_pbar = tqdm(desc="Simulations", unit="sim", leave=True, position=0)
    pbar = tqdm(total=50, desc="Accepted", unit="draw", leave=True, position=1)

    while len(kept) < 50:
        attempts += 1
        # theta = draw_near_true(rng, true_theta.copy())
        theta = true_theta
        traj = simulators.FHN_model(initial_value=np.zeros(2), theta=theta, timestep=d, number_of_samples=n)
        sim_pbar.update(1)
        if not np.all(np.isfinite(traj)):
            continue
        try:
            dist = dist_calc.eval(traj)
        except Exception:
            continue
        if not np.isfinite(dist):
            continue
        if dist <= threshold:
            kept.append(theta)
            kept_dists.append(float(dist))
            pbar.update(1)
            pbar.set_postfix({"attempts": attempts, "dist": f"{dist:.5f}"})
            pbar.write(f"kept {len(kept)}/50 | attempts={attempts} | dist={dist:.5f} | theta={theta}")

    pbar.close()
    sim_pbar.close()

    kept = np.array(kept)
    kept_dists = np.array(kept_dists)

    # Summary
    print("\nDone.")
    print(f"Attempts: {attempts}")
    print(f"Acceptance rate: {len(kept) / attempts:.3f}")
    print(f"Distances: min={kept_dists.min():.5f}, median={np.median(kept_dists):.5f}, max={kept_dists.max():.5f}")

    # Save
    out = np.column_stack([kept, kept_dists])
    header = "epsilon,gamma,beta,sigma,distance"
    np.savetxt("near_true_accepts.csv", out, delimiter=",", header=header, comments="")
    print("Saved accepted parameters to near_true_accepts.csv")


if __name__ == "__main__":
    main()
