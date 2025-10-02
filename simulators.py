import numpy as np
from numba import njit

# trajectory simulation of FHN model using Strang splitting scheme
# returns time-series for the data
@njit(fastmath=True)
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
    delta_half = delta/2
    kappa = 4*gamma/epsilon - 1
    sqrt_kappa = np.sqrt(kappa)
    arg1 = sqrt_kappa * delta_half 
    cos_term1 = np.cos(arg1)
    sin_term1 = np.sin(arg1)
    E = np.exp(-delta_half) * np.array([
        [cos_term1 + sin_term1/sqrt_kappa, -2*sin_term1 / (sqrt_kappa*epsilon)],
        [2 * gamma * sin_term1/sqrt_kappa, cos_term1 - sin_term1/sqrt_kappa]
    ])
    # C(delta)
    coeff = sigma**2 * np.exp(-delta) / kappa
    arg2 = sqrt_kappa * delta
    cos_term2 = np.cos(arg2)
    sin_term2 = np.sin(arg2)
    term1 = -4*gamma/epsilon + kappa*np.exp(delta) # re-ocurring term in below
    c11 = (coeff / (2*epsilon*gamma)) * (term1 + cos_term2 - sqrt_kappa*sin_term2)
    c12 = c21 = (coeff / epsilon) * (cos_term2 - 1)
    c22 = (coeff /2) * (cos_term2 + sqrt_kappa*sin_term2 + term1)
    C = np.array([[c11, c12], [c21, c22]])

    # non-linear ODE solution
    t = delta/2
    exp_term = np.exp(-2 * t / epsilon)
    one_minus_exp = 1 - exp_term
    beta_t = beta * t
    def h(v_prev, u_prev):
        sqr_arg = exp_term + (v_prev**2)*one_minus_exp
        v_next = v_prev / np.sqrt(sqr_arg)
        u_next = beta_t + u_prev
        return v_next, u_next

    # simulating trajectory
    # X[i] = [v_i, u_i]
    X = np.empty((N, 2))
    X[0] = X0
    L = np.linalg.cholesky(C)
    a = np.empty(2)
    std_noise = np.empty(2)
    noise = np.empty(2)
    for i in range(N-1):
        a0, a1 = h(X[i, 0], X[i, 1])
        a[0] = a0
        a[1] = a1
        std_noise[0] = np.random.normal()
        std_noise[1] = np.random.normal()
        noise[:] = L @ std_noise
        b = E@a + noise

        v_next, u_next = h(b[0], b[1])
        X[i+1, 0] = v_next
        X[i+1, 1] = u_next
    return X[:,0]  # return only the voltage time-series

@njit(fastmath=True)
def OU_model(initial_value, theta, timestep, number_of_samples):
    ''' dX = (00 - θ1 X)dt + θ2dW'''
    X = np.empty(number_of_samples)
    temp = theta[1]*timestep
    X[0] = initial_value
    var = theta[2]**2 * (1 - np.exp(-2*temp) ) / (2*theta[1])
    sd = np.sqrt(var)

    for i in range(number_of_samples):
        mean = theta[0]/theta[1] + (X[i] - theta[0]/theta[1]) * np.exp(-temp)
        X[i+1] = np.random.normal(mean, sd)
    return X
    