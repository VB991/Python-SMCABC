import distances
import simulators
import numpy as np
import matplotlib.pyplot as plt

data = simulators.FHN_model(initial_value = np.zeros(2), theta = [0.1, 1.5, 0.8, 0.3], timestep=0.08, number_of_samples = 625)
traj1 = simulators.FHN_model(initial_value = np.zeros(2), theta = [0.1, 1.5, 0.8, 0.3], timestep=0.08, number_of_samples = 625)
traj2 = simulators.FHN_model(initial_value = np.zeros(2), theta = [0.4, 3.0, 1.6, 0.05], timestep=0.08, number_of_samples = 625)
dist_calc = distances.CalculateModelBasedDistance(data, 0.08, span=51)
plt.plot(data)
plt.plot(traj1, linestyle="dotted")
plt.show()
print(dist_calc.eval(traj1))
plt.plot(data)
plt.plot(traj2, linestyle="dotted")
plt.show()
print(dist_calc.eval(traj2))