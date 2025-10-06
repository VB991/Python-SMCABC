"""Self-contained ABC demo with a deterministic 4-parameter ODE simulator.

- Simulator: 1D ODE dx/dt = a*x + b*sin(c*t) + d (deterministic Euler)
- Prior: Independent uniform box over (a,b,c,d)
- Distance: Integrated Absolute Error (IAE) between observed and simulated trajectory
- Inference: Calls SMCABC.sample_posterior with 1000 particles, budget 100000, threshold_percentile=0.5

Run:
  python main2.py
"""

from __future__ import annotations

import numpy as np
from typing import Sequence

import SMCABC
from distances import CalculateDistance


# ----------------------------
# Deterministic ODE simulator
# ----------------------------
def ode_simulator(initial_value: np.ndarray | float,
                  theta: Sequence[float],
                  timestep: float,
                  number_of_samples: int) -> np.ndarray:
    """Deterministic 1D ODE using explicit Euler.

    dx/dt = a*x + b*sin(c*t) + d

    Args:
        initial_value: scalar or array-like with one element specifying x(0)
        theta: [a, b, c, d]
        timestep: dt
        number_of_samples: length of the returned trajectory
    Returns:
        1D numpy array of length `number_of_samples`
    """
    # Robustly coerce theta to a flat length-4 vector of floats
    th = np.asarray(theta, dtype=np.float64)
    if th.ndim > 1:
        th = th.reshape(-1)
    if th.size != 4:
        raise ValueError(f"theta must have 4 elements, got shape {np.shape(theta)}")
    a, b, c, d = th.tolist()
    dt = float(timestep)
    n = int(number_of_samples)

    x0 = float(np.atleast_1d(initial_value)[0])
    t = 0.0
    traj = np.empty(n, dtype=np.float32)
    x = x0
    traj[0] = x
    for i in range(1, n):
        # Euler step
        u = b * np.sin(c * t) + d
        x = x + dt * (a * x + u)
        t += dt
        traj[i] = x
    return traj


# ----------------------------
# Uniform prior over parameters
# ----------------------------
class UniformBox4D:
    """Independent uniform prior over a 4D hyper-rectangle.

    bounds = [(a_low,a_high), (b_low,b_high), (c_low,c_high), (d_low,d_high)]
    """
    def __init__(self, bounds: Sequence[Sequence[float]]):
        self.bounds = np.asarray(bounds, dtype=float)
        if self.bounds.shape != (4, 2):
            raise ValueError("Expected bounds shape (4,2) for 4 parameters")
        self.lows = self.bounds[:, 0]
        self.highs = self.bounds[:, 1]
        if np.any(self.highs <= self.lows):
            raise ValueError("Each upper bound must be > lower bound")
        self.volume = float(np.prod(self.highs - self.lows))

    def rvs(self, size: int = 1, random_state: int | None = None) -> np.ndarray:
        rng = np.random.default_rng(random_state)
        return rng.uniform(self.lows, self.highs, size=(size, 4)).astype(np.float32)

    def pdf(self, x: np.ndarray | Sequence[float]) -> np.ndarray | float:
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            inside = np.all((x >= self.lows) & (x <= self.highs))
            return (1.0 / self.volume) if inside else 0.0
        elif x.ndim == 2:
            inside = np.all((x >= self.lows) & (x <= self.highs), axis=1)
            out = np.zeros(x.shape[0], dtype=float)
            out[inside] = 1.0 / self.volume
            return out
        else:
            raise ValueError("x must be a 1D or 2D array")


# ----------------------------
# Distance: Integrated Absolute Error
# ----------------------------
class IntegratedAbsoluteErrorDistance(CalculateDistance):
    """Distance = integral over time of |y_real(t) - y_sim(t)|."""
    def __init__(self, real_trajectory: np.ndarray, timestep: float):
        super().__init__(np.asarray(real_trajectory, dtype=np.float32), float(timestep))

    def _summarise(self, trajectory: np.ndarray) -> np.ndarray:
        return np.asarray(trajectory, dtype=np.float32)

    def _calculate_summaries_distance(self, simulation_summary: np.ndarray) -> float:
        if simulation_summary.shape != self.summary.shape:
            raise ValueError("Observed and simulated trajectories must have the same shape")
        diff = np.abs(self.summary - simulation_summary)
        # Use trapezoidal rule with uniform spacing dt
        return float(np.trapz(diff, dx=self.timestep))


def main():
    # --- Configuration ---
    dt = 0.05
    n = 600

    # Prior over parameters [a, b, c, d]
    prior = UniformBox4D(bounds=[
        [-1.2, -0.1],  # a: stable linear dynamics
        [0.0, 2.0],    # b: forcing amplitude
        [0.5, 2.0],    # c: forcing frequency
        [-1.0, 1.0],   # d: constant input
    ])

    # Simulate observed data at a fixed "true" parameter
    true_theta = np.array([-0.6, 1.2, 1.1, 0.2], dtype=np.float32)
    x0 = np.array([0.0], dtype=np.float32)
    data = ode_simulator(x0, true_theta, dt, n)

    # Distance calculator
    dist_calc = IntegratedAbsoluteErrorDistance(real_trajectory=data, timestep=dt)

    # ABC-SMC posterior sampling
    particles, weights = SMCABC.sample_posterior(
        data=data,
        initial_value=x0,
        timestep=dt,
        threshold_percentile=0.2,
        prior=prior,
        model_simulator=ode_simulator,
        distance_calculator=dist_calc,
        num_samples=100,
        simulation_budget=2500,
    )

    # Report results
    mean = np.average(particles, axis=0, weights=weights)
    cov = np.cov(particles.T, aweights=weights)
    print("True theta:", true_theta)
    print("Posterior mean:", mean)
    print("Posterior diag(cov):", np.diag(cov))


if __name__ == "__main__":
    # On Windows, this guard is required for multiprocessing used by SMCABC
    main()
