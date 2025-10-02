import distances
import simulators
import numpy as np
import matplotlib.pyplot as plt

# Change here to switch model type
model = "FHN"

# Subsampling parameters as in main.py
d = 0.08
n = 2500
T = n*d


# Simulate all trajectories at the real (true) theta
FHNtrue_theta = np.array([0.1, 1.5, 0.8, 0.3], dtype=float)
OUtrue_theta = np.array([0.0, 1.0, 1.0], dtype=float)
true_theta = FHNtrue_theta if model == "FHN" else OUtrue_theta

# Load and simulate trajectories
FHNdata = np.loadtxt("observation.txt")[0:int(T/0.0001):int(d/0.0001)]
OUdata = simulators.OU_model(0.0, OUtrue_theta, d, n)
data = FHNdata if model == "FHN" else OUdata

# How many simulated trajectories to compare
num_trajs = 5  # change this to desired count

trajectories = [("data", data)]
for i in range(num_trajs):
    if model == "FHN":
        traj = simulators.FHN_model(initial_value=np.zeros(2), theta=true_theta, timestep=d, number_of_samples=n)
    elif model == "OU":
        traj = simulators.OU_model(0.0, [0.0, 1.0, 1.0], d, n)
    else:
        raise ValueError("Invalid model type specified")
    trajectories.append((f"traj{i+1}", traj))

# Build distance calculator on the real data to reuse its KDE grid and smoothing
dist_calc = distances.CalculateModelBasedDistance(real_trajectory=data, timestep=d)
(ends_real, _kde_ignored, freqs_ref, spec_real) = dist_calc.summary
grid_ref = dist_calc.grid
grid_spacing = dist_calc.grid_spacing
pdf_real = dist_calc.pdf

# Summaries for each trajectory (KDE objects and smoothed spectra)
summaries = {}
max_left_pts = 0
max_right_pts = 0

for name, traj in trajectories:
    if name == "data":
        summaries[name] = {
            "ends": ends_real,
            "kde": None,  # already have evaluated pdf for data
            "spec": spec_real,
        }
        continue
    ends, kde, freqs, spec = dist_calc._summarise(traj)
    summaries[name] = {
        "ends": ends,
        "kde": kde,
        "spec": spec,
        "freqs": freqs,
    }
    # Determine how much to extend ref grid to cover this support
    left_diff = float(ends_real[0] - ends[0])
    right_diff = float(ends[1] - ends_real[1])
    n_left = int(np.ceil(max(0.0, left_diff) / grid_spacing))
    n_right = int(np.ceil(max(0.0, right_diff) / grid_spacing))
    if n_left > max_left_pts:
        max_left_pts = n_left
    if n_right > max_right_pts:
        max_right_pts = n_right

# Build common grid for PDF overlays (extends real data grid to cover all)
new_start = grid_ref[0] - max_left_pts * grid_spacing
total_len = max_left_pts + grid_ref.size + max_right_pts
grid_all = new_start + grid_spacing * np.arange(total_len, dtype=float)

# Evaluate / pad PDFs for each trajectory on common grid
pdfs = {}
pdfs["data"] = np.concatenate(
    (
        np.zeros(max_left_pts, dtype=float),
        pdf_real,
        np.zeros(max_right_pts, dtype=float),
    )
)
for name, _traj in trajectories:
    if name == "data":
        continue
    kde = summaries[name]["kde"]
    pdfs[name] = kde.evaluate(grid_all)

# Plot PDF overlays
plt.figure()
for name, _ in trajectories:
    ls = "-" if name == "data" else ":"  # dotted for simulated PDFs
    lw = 2.5 if name == "data" else 1.0   # thicker for real trajectory
    plt.plot(grid_all, pdfs[name], label=name, linestyle=ls, linewidth=lw)
plt.title("Estimated PDF overlays")
plt.xlabel("Value")
plt.ylabel("Density")
plt.legend()
plt.tight_layout()
plt.show()

# Plot spectral density overlays
plt.figure()
plt.plot(freqs_ref, spec_real, label="data", linewidth=2.5, linestyle="-")  # thicker for real trajectory
for name, _traj in trajectories:
    if name == "data":
        continue
    freqs = summaries[name].get("freqs", freqs_ref)
    spec = summaries[name]["spec"]
    if not np.allclose(freqs, freqs_ref):
        # If ever mismatched, interpolate onto reference grid
        spec = np.interp(freqs_ref, freqs, spec)
    plt.plot(freqs_ref, spec, label=name, linewidth=1.0, linestyle=":")
plt.title("Smoothed spectral density overlays")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Spectral density")
plt.legend()
plt.tight_layout()
plt.show()
