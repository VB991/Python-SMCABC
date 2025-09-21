import distances
import simulators
import numpy as np

data = simulators.FHN_model(initial_value = np.zeros(2), theta = [0.1, 1.5, 0.8, 0.3], timestep=0.08, number_of_samples = 625)
traj1 = simulators.FHN_model(initial_value = np.zeros(2), theta = [0.1, 1.5, 0.8, 0.3], timestep=0.08, number_of_samples = 625)
traj2 = simulators.FHN_model(initial_value = np.zeros(2), theta = [0.4, 5.0, 5.0, 0.05], timestep=0.08, number_of_samples = 625)
dist_calc = distances.CalculateModelBasedDistance(data, 0.08)
print(dist_calc.eval(traj1))
print(dist_calc.eval(traj2))