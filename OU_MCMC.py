import numpy as np
import tqdm
from numba import njit
import concurrent.futures


@njit(fastmath=True)
def _ou_loglik(x: np.ndarray, dt: float, params: np.ndarray) -> float:
    """Numba-accelerated OU transition log-likelihood.

    x: observed 1D time series of length T
    theta: array([mu, theta, sigma])
    """
    n = x.shape[0]
    mu = params[0]
    theta = params[1]
    sigma = params[2]
    rho = np.exp(-theta * dt)
    var = (sigma * sigma) * (1.0 - rho * rho) / (2.0 * theta)
    logpdf = 0.0
    for i in range(n - 1):
        mean = mu + (x[i] - mu) * rho
        diff = x[i + 1] - mean
        logpdf += -0.5 * (np.log(2.0 * np.pi * var) + (diff * diff) / var)
    return logpdf


def OU_MCMC(theta, prior, dt, observations, num_iterations):
    """Run Metropolis-Hastings for OU: dX = (mu - theta X) dt + sigma dW.

    Args:
        theta: initial parameter (mu, theta, sigma)
        prior: prior object with .pdf(x)
        dt: time step between observations
        observations: observed 1D time series (array-like)
        num_iterations: number of MCMC iterations
    Returns:
        np.ndarray of shape (num_iterations, 3)
    """
    x = np.asarray(observations, dtype=np.float64).ravel()
    theta = np.asarray(theta, dtype=np.float64)

    p = len(theta)
    eps = 1e-8
    s_d = (2.4 ** 2) / p

    cov = np.diag([0.1] * p).astype(np.float64)
    L = np.linalg.cholesky(cov)
    chain = np.zeros((num_iterations, p), dtype=np.float64)
    chain[0] = theta
    lik = _ou_loglik(x, float(dt), theta)

    rng = np.random.default_rng()
    for i in tqdm.tqdm(range(num_iterations - 1)):

        z = np.random.standard_normal(p)
        theta_new = theta + L @ z
        lik_new = _ou_loglik(x, float(dt), theta_new)

        log_prob_old = float(np.log(prior.pdf(theta)))
        log_prob_new = float(np.log(prior.pdf(theta_new)))

        alpha = lik_new + log_prob_new - (lik + log_prob_old)
        if np.log(np.random.uniform()) < alpha:
            theta = theta_new
            lik = lik_new

        chain[i + 1] = theta

        if i > 10000 and i % 1000 == 0:
            cov = s_d * np.cov(chain[: i + 1].T) + s_d * eps * np.eye(p)
            L = np.linalg.cholesky(cov)

    return chain
