import distances
import simulators
import numpy as np
import matplotlib.pyplot as plt

d = 0.08
n = 625
data = simulators.FHN_model(initial_value = np.zeros(2), theta = [0.1, 1.5, 0.8, 0.3], timestep=d, number_of_samples = n)
traj1 = simulators.FHN_model(initial_value = np.zeros(2), theta = [0.1, 1.5, 0.8, 0.3], timestep=d, number_of_samples = n)
traj2 = simulators.FHN_model(initial_value = np.zeros(2), theta = [0.1, 1.6, 0.8, 0.4], timestep=d, number_of_samples = n)
dist_calc = distances.CalculateModelBasedDistance(data, d)

plt.plot(data)
plt.plot(traj1, linestyle="dotted")
plt.show()
print(dist_calc.eval(traj1))
plt.plot(data)
plt.plot(traj2, linestyle="dotted")
plt.show()
print(dist_calc.eval(traj2))
























