"""Train a PEN-based summary network on FHN trajectories and sanity-check summaries."""
import numpy as np
import torch

from distances import CalculatePENDistance
import simulators

FHN = simulators.FHN_model.py_func

# ----------------------------
# Configuration
# ----------------------------
SEQ_LEN = 400
MARKOV_ORDER = 1
PARAM_DIM = 4
TIMESTEP = 0.05
NUM_TRAIN = 1000
BATCH = 32
EPOCHS = 25
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

rng = np.random.default_rng()

PARAM_BOUNDS = np.array([
    [0.05, 0.5],   # epsilon
    [0.5, 3.0],    # gamma
    [0.2, 1.5],    # beta
    [0.05, 0.6],   # sigma
], dtype=np.float32)


def sample_theta(n):
    lows, highs = PARAM_BOUNDS[:, 0], PARAM_BOUNDS[:, 1]
    return rng.uniform(lows, highs, size=(n, PARAM_DIM)).astype(np.float32)


# ----------------------------
# Generate training set
# ----------------------------
train_thetas = sample_theta(NUM_TRAIN)
train_trajs = []
for theta in train_thetas:
    traj = FHN(np.zeros(2, dtype=np.float32), theta, TIMESTEP, SEQ_LEN).astype(np.float32)
    train_trajs.append(traj)
train_trajs = np.stack(train_trajs)

# Use first training trajectory as real trajectory
real_trajectory = train_trajs[0]
train_trajs_rest = train_trajs  # All trajectories used for training

# Instantiate and train CalculatePENDistance
pen_distance = CalculatePENDistance(
    real_trajectory=real_trajectory,
    timestep=TIMESTEP,
    parameter_dim=PARAM_DIM,
    training_data_x=train_trajs_rest,
    training_data_params=train_thetas,
    markov_order=MARKOV_ORDER,
    batch_size=BATCH,
    num_epochs=EPOCHS,
    device=str(DEVICE),
)

# ----------------------------
# Sanity check: same theta, different trajectories
# ----------------------------
theta_test = sample_theta(1)[0]
traj1 = FHN(np.zeros(2, dtype=np.float32), theta_test, TIMESTEP, SEQ_LEN)
traj2 = FHN(np.zeros(2, dtype=np.float32), theta_test, TIMESTEP, SEQ_LEN)

summary1 = pen_distance._summarise(traj1)
summary2 = pen_distance._summarise(traj2)

diff = torch.linalg.norm(summary1 - summary2).item()

print("θ used for sanity check:", theta_test)
print("Summary 1:", summary1.numpy())
print("Summary 2:", summary2.numpy())
print(f"Euclidean distance between summaries: {diff:.6f}")


# Check summaries for different trajectories
theta_far = sample_theta(2)
traj_far_1 = FHN(np.zeros(2, dtype=np.float32), theta_far[0], TIMESTEP, SEQ_LEN)
traj_far_2 = FHN(np.zeros(2, dtype=np.float32), theta_far[1], TIMESTEP, SEQ_LEN)

summary_far_1 = pen_distance._summarise(traj_far_1)
summary_far_2 = pen_distance._summarise(traj_far_2)

diff_far = torch.linalg.norm(summary_far_1 - summary_far_2).item()

print("\nθ values for separation check:")
print("θ1:", theta_far[0])
print("θ2:", theta_far[1])
print("Summary θ1:", summary_far_1.numpy())
print("Summary θ2:", summary_far_2.numpy())
print(f"Euclidean distance between summaries (different θ): {diff_far:.6f}")
