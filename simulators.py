import numpy as np
from numba import njit

# trajectory simulation of FHN model using Strang splitting scheme
# returns time-series for the data
@njit
def FHN_model(initial_value, theta, timestep, number_of_samples):
    # parameters for the FitzHugh-Nagumo model
    # theta = [epsilon, gamma, beta, sigma]
    X0 = initial_value
    N = number_of_samples
    delta = timestep
    epsilon = theta[0]
    gamma = theta[1]
    beta = theta[2]
    sigma = theta[3]

    # parts for the linear SDE
    # E(delta)
    kappa = 4 * gamma / epsilon - 1
    sqrt_kappa = np.sqrt(kappa)
    arg1 = sqrt_kappa * delta / 2 
    cos_term1 = np.cos(arg1)
    sin_term1 = np.sin(arg1)
    E = np.exp(-delta / 2) * np.array([
        [cos_term1 + sin_term1 / sqrt_kappa, -2 * sin_term1 / (sqrt_kappa * epsilon)],
        [2 * gamma * sin_term1 / sqrt_kappa, cos_term1 - sin_term1 / sqrt_kappa]
    ])
    # C(delta)
    coeff = sigma**2 * np.exp(-delta) / kappa
    arg2 = sqrt_kappa * delta
    cos_term2 = np.cos(arg2)
    sin_term2 = np.sin(arg2)
    term1 = -4*gamma/epsilon + kappa*np.exp(delta) # reocurring term in below
    c11 = (coeff / (2*epsilon*gamma)) * (term1 + cos_term2 - sqrt_kappa*sin_term2)
    c12 = c21 = (coeff / epsilon) * (cos_term2 - 1)
    c22 = (coeff /2) * (cos_term2 + sqrt_kappa*sin_term2 + term1)
    C = np.array([[c11, c12], [c21, c22]])
    # non-linear ODE solution
    def h(x, t):
        v = x[0]
        u = x[1]
        exp_term = np.exp(-2 * t / epsilon)
        return np.array([
            v / np.sqrt(exp_term + v**2 * (1 - exp_term)),
            beta*t + u
        ])

    # simulating trajectory
    # X[i] = [v_i, u_i]
    X = np.zeros((N, 2))
    X[0] = X0
    L = np.linalg.cholesky(C)
    for i in range(N-1):
        a = h(X[i], delta/2)
        noise = L@np.random.normal(0,1,2)
        b = E@a + noise
        X[i+1] = h(b, delta/2)
    return X[:,0]  # return only the voltage time-series