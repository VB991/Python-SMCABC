from abc import ABC, abstractmethod
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from KDEpy.FFTKDE import FFTKDE
from scipy import signal, integrate
from concurrent.futures import ProcessPoolExecutor
from numba import njit

# Globals used only by multiprocessing workers for PEN data simulation
_PEN_SIM_X0 = None
_PEN_SIM_DT = None
_PEN_SIM_N = None
_PEN_SIM_MODEL = None

def _pen_worker_init(x0, timestep, num_samples, model_simulator):
    """Initializer for worker processes; stores shared args in globals.

    Using globals avoids re‑serializing large immutable inputs for each task
    and keeps the worker function picklable on Windows (spawn start method).
    """
    global _PEN_SIM_X0, _PEN_SIM_DT, _PEN_SIM_N, _PEN_SIM_MODEL
    _PEN_SIM_X0 = x0
    _PEN_SIM_DT = timestep
    _PEN_SIM_N = num_samples
    _PEN_SIM_MODEL = model_simulator


def _pen_sim_worker(theta_batch):
    """Simulate a batch of training trajectories for a batch of parameter vectors."""
    out = []
    for theta in theta_batch:
        traj = _PEN_SIM_MODEL(_PEN_SIM_X0, theta, _PEN_SIM_DT, _PEN_SIM_N)
        out.append(np.asarray(traj, dtype=np.float32))
    return out


@njit(fastmath=True, cache=True)
def _pen_feedforward_jit(
    trajectory,
    k,
    W_e_in, b_e_in,
    W_e_hidden, b_e_hidden,
    W_e_out, b_e_out,
    W_h_in, b_h_in,
    W_h1, b_h1,
    W_h2, b_h2,
    W_h_out, b_h_out,
):
    n = trajectory.shape[0]
    nb = n - k

    enc1_out = W_e_in.shape[0]
    enc2_out = W_e_hidden.shape[0]
    enc3_out = W_e_out.shape[0]

    # Temp buffers
    h1 = np.empty(enc1_out, dtype=np.float32)
    h2 = np.empty(enc2_out, dtype=np.float32)
    h3 = np.empty(enc3_out, dtype=np.float32)
    aggregated = np.zeros(enc3_out, dtype=np.float32)

    # Encode blocks and sum
    for i in range(nb):
        # x_block = trajectory[i:i+k+1]
        # First encoder layer: y = ReLU(b + W @ x)
        for j in range(enc1_out):
            s = b_e_in[j]
            for t in range(W_e_in.shape[1]):
                s += W_e_in[j, t] * trajectory[i + t]
            h1[j] = s if s > 0.0 else 0.0

        # Second encoder layer
        for j in range(enc2_out):
            s = b_e_hidden[j]
            for t in range(W_e_hidden.shape[1]):
                s += W_e_hidden[j, t] * h1[t]
            h2[j] = s if s > 0.0 else 0.0

        # Third encoder layer
        for j in range(enc3_out):
            s = b_e_out[j]
            for t in range(W_e_out.shape[1]):
                s += W_e_out[j, t] * h2[t]
            val = s if s > 0.0 else 0.0
            h3[j] = val
            aggregated[j] += val

    # Build head input [first_k, aggregated]
    xlen = k + enc3_out
    x_full = np.empty(xlen, dtype=np.float32)
    for i in range(k):
        x_full[i] = trajectory[i]
    for j in range(enc3_out):
        x_full[k + j] = aggregated[j]

    # Head layer 1
    h1_len = W_h_in.shape[0]
    hh1 = np.empty(h1_len, dtype=np.float32)
    for j in range(h1_len):
        s = b_h_in[j]
        for t in range(W_h_in.shape[1]):
            s += W_h_in[j, t] * x_full[t]
        hh1[j] = s if s > 0.0 else 0.0

    # Head layer 2
    h2_len = W_h1.shape[0]
    hh2 = np.empty(h2_len, dtype=np.float32)
    for j in range(h2_len):
        s = b_h1[j]
        for t in range(W_h1.shape[1]):
            s += W_h1[j, t] * hh1[t]
        hh2[j] = s if s > 0.0 else 0.0

    # Head layer 3
    h3_len = W_h2.shape[0]
    hh3 = np.empty(h3_len, dtype=np.float32)
    for j in range(h3_len):
        s = b_h2[j]
        for t in range(W_h2.shape[1]):
            s += W_h2[j, t] * hh2[t]
        hh3[j] = s if s > 0.0 else 0.0

    # Output layer (linear)
    out_len = W_h_out.shape[0]
    out = np.empty(out_len, dtype=np.float32)
    for j in range(out_len):
        s = b_h_out[j]
        for t in range(W_h_out.shape[1]):
            s += W_h_out[j, t] * hh3[t]
        out[j] = s

    return out




class CalculateDistance(ABC):
    """Abstract base class for trajectory distance calculators."""
    @abstractmethod
    def _summarise(trajectory):
        pass

    @abstractmethod
    def _calculate_summaries_distance(self, simulation_summary):
        pass

    def __init__(self, real_trajectory, timestep):
        self.timestep = timestep
        self.summary = self._summarise(real_trajectory)

    def eval(self, simulation_trajectory):
        sim_summary = self._summarise(simulation_trajectory)
        return self._calculate_summaries_distance(sim_summary)








class CalculateModelBasedDistance(CalculateDistance):
    def __init__(self, real_trajectory, timestep, spans: list[int] = None):
        """Calculate trajectory distance using estimated density and spectral density as summaries.

        Args:
            real_trajectory (_type_): Real data
            timestep (_type_): Timestep of data
            span (_type_): Span used for modified boxcar kernel when smoothing periodogram.
        """

        # Intialise kernel for smoothing periodogram
        if spans is None:
            n = round(len(real_trajectory)*timestep*0.3)
            n = n if n%2==1 else n+1
            m = int(max((n-1)/4 + 1, 2))
            spans = [n,m]
        else:
            for span in spans:
                if span < 3 or span%2 == 0:
                    raise ValueError("Spans of periodogram smoothing kernels must be an odd integer, at least 3")
        ker = np.array([1.0])
        for span in spans:
            box = np.ones(span, float)
            box[0] = box[-1] = 0.50
            box /= (span-1)
            ker = np.convolve(box, ker, mode="full")
        ker /= np.sum(ker)
        self.smooth_ker = ker

        super().__init__(real_trajectory, timestep)
        ends, _, frequencies, smooth_spectral_density = self.summary

        # Build a fixed grid of the real-data support, and evaluate pdf_real
        self.grid = np.linspace(ends[0], ends[1], 1024)
        self.grid_spacing = float(self.grid[1] - self.grid[0])
        kde_real = FFTKDE(kernel="gaussian", bw="silverman")
        kde_real.fit(real_trajectory)
        self.pdf = kde_real.evaluate(self.grid)

        # Omit unpicklable kde object (to allow for multiprocessing)
        self.summary = (ends, None, frequencies, smooth_spectral_density)

    def _spectrum(self, trajectory, timestep):
        # ----- Smoothed periodogram ------
        fs = 1.0 / timestep
        window = signal.windows.tukey(M=trajectory.size, alpha=0.2)
        frequencies, spectral_density = signal.welch(
            trajectory,
            fs=fs,
            window=window,
            detrend="linear",
            return_onesided=True,
            scaling="density",
        )

        spectral_density[0] = spectral_density[1:4].mean()
        
        mask = frequencies <= 2

        # Smooth with modified boxcar kernels (ends half the weight)
        pad = len(self.smooth_ker) // 2
        smooth_spec = np.r_[spectral_density[pad:0:-1], spectral_density, spectral_density[-2:-pad-2:-1]]
        spectral_density = signal.fftconvolve(smooth_spec, self.smooth_ker, mode="same")[pad:pad+len(spectral_density)]

        return frequencies[mask], spectral_density[mask]

    def _summarise(self, trajectory):
        # ------ Calculate summary of trajectory --------

        # KDE object for estimated density, support for KDE
        kde = FFTKDE(kernel="gaussian", bw="silverman")
        kde.fit(trajectory)
        padding = 2*trajectory.std()    # ensure bulk of pdf is contained
        kde_support_ends = (trajectory.min()-padding, trajectory.max()+padding) 

        # Smoothed periodogram
        frequencies, spectral_density = self._spectrum(trajectory, self.timestep)

        # ------ Return: Endpoints for pdf support, kde object, frequencies, and spectral density at frequencies ------
        return (kde_support_ends, kde, frequencies, spectral_density)

    def _calculate_summaries_distance(self, simulation_summary):
        ends1, _, frequencies1, spectral_density1 = self.summary
        ends2, kde2, frequencies2, spectral_density2 = simulation_summary

        if not np.allclose(frequencies1, frequencies2):
            raise ValueError("Periodogram frequencies do not match. Ensure that the time duration of both trajectories is the same.")
        else:
            freqs = frequencies1

        # Evaluate KDEs over a common grid. Build a single uniform grid that
        # extends the real-data grid by integer steps on each side to cover the
        # simulated support. This guarantees constant spacing required by KDEpy.
        left_diff = float(ends1[0] - ends2[0])
        right_diff = float(ends2[1] - ends1[1])

        n_left_points = int(np.ceil(max(0.0, left_diff) / self.grid_spacing))
        n_right_points = int(np.ceil(max(0.0, right_diff) / self.grid_spacing))

        new_start = self.grid[0] - n_left_points * self.grid_spacing
        total_len = n_left_points + self.grid.size + n_right_points
        grid = new_start + self.grid_spacing * np.arange(total_len, dtype=float)

        sim_pdf = kde2.evaluate(grid)
        # Pad real PDF with zeros to match grid length
        real_pdf = np.concatenate((
            np.zeros(n_left_points, dtype=float),
            self.pdf,
            np.zeros(n_right_points, dtype=float),
        ))
    

        # Compute integrated absolute differences, combine via IAE1 + alpha*IAE2
        pdf_distance = integrate.trapezoid(np.abs(real_pdf - sim_pdf), grid)
        spectral_density_distance = integrate.trapezoid(y = np.abs(spectral_density1 - spectral_density2), x = freqs)
        alpha = integrate.trapezoid(y = np.abs(spectral_density1), x = freqs)

        return spectral_density_distance + alpha*pdf_distance





    
class CalculatePENDistance(CalculateDistance):
    def __init__(self):
        self.layers = None  # Dictionary containing PEN weights and biases
        self.k = None       # Markov order

    # ----- 
    def _simulate_training_trajs(self, training_thetas, traj_initial_value, model_simulator, timestep, num_samples):
        print("Simulating training trajectories...")
        thetas = np.asarray(training_thetas)
        if thetas.ndim == 1:
            thetas = thetas.reshape(1, -1)

        n_tasks = int(thetas.shape[0])

        # Progress bar
        from tqdm import tqdm

        with ProcessPoolExecutor(
            max_workers=None,
            initializer=_pen_worker_init,
            initargs=(traj_initial_value, timestep, num_samples, model_simulator),
        ) as ex:
            pbar = tqdm(total=n_tasks, desc="Training sims")
            # Create at most 1000 futures by batching thetas
            max_futures = 1000
            n_batches = n_tasks if n_tasks < max_futures else max_futures
            batch_size = (n_tasks + n_batches - 1) // n_batches
            batches = [thetas[i:i+batch_size] for i in range(0, n_tasks, batch_size)]

            results = []
            for batch_res in ex.map(_pen_sim_worker, batches):
                # batch_res is a list of trajectories
                results.extend(batch_res)
                pbar.update(len(batch_res))
            pbar.close()
        training_trajs = np.asarray(results, dtype=np.float32)

        print("finished!")
        return training_trajs


    # ----- Alternative Constructor -----
    def create_and_train_PEN(
            self, 
            model_simulator: callable, training_thetas, traj_initial_value, real_trajectory, timestep, dimension_weights, 
            markov_order = 1, 
            device_name = "cpu", batch_size=32, num_epochs=25,
            early_stopping_patience=10, early_stopping_loss_drop=0.0, validation_split=0.2,
            ):
        
        # Lazy imports to prevent torch dependency in instances of this class
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from torch.utils.data import DataLoader, TensorDataset


        #  ------ Define internal PEN pytorch neural network: only used for construction and training ------
        class PEN(nn.Module):
            '''Partially Exchangeable Network for learning summary statistics for Markov process'''
            def __init__(self, parameter_dim, markov_order=1):
                super().__init__()
                self.k = markov_order

                # Layers for PEN inner function
                self.encoder_in = nn.Linear(self.k+1, 100)
                self.encoder_hidden = nn.Linear(100,50)
                self.encoder_out = nn.Linear(50,10)

                # Layers for PEN outer function
                self.head_in = nn.Linear(10+self.k, 100)
                self.head_hidden1 = nn.Linear(100, 100)
                self.head_hidden2 = nn.Linear(100, 50)
                self.head_out = nn.Linear(50, parameter_dim)

            def _apply_encoder(self, x):
                '''Apply representation layer to input time series'''
                # Convert unbatched input to a batch of size 1.
                if x.dim() == 1:
                    x = x.unsqueeze(0)
                elif x.dim() != 2:
                    raise ValueError("Expected batched or unbatched 1D input")

                # Ensure sufficiently sized input
                if x.size(1) < self.k + 1:
                    raise ValueError(
                        f"Input length ({x.size(1)}) must be at least markov_order+1 ({self.k + 1})."
                    )
                
                # Create the subsequence blocks for encoding
                blocks = x.unfold(dimension=1, size=self.k + 1, step=1)
                batch_size, num_blocks, _ = blocks.shape
                # Flatten windows to (batch_size * num_blocks, k+1) for the encoder
                blocks = blocks.reshape(-1, self.k + 1)

                # Encode each block and reshape back to (batch_size, num_blocks, encoded_dim)
                encoded_blocks = nn.functional.relu(self.encoder_in(blocks))
                encoded_blocks = nn.functional.relu(self.encoder_hidden(encoded_blocks))
                encoded_blocks = nn.functional.relu(self.encoder_out(encoded_blocks))
                encoded_blocks = encoded_blocks.reshape(batch_size, num_blocks, encoded_blocks.size(-1))

                # Sum the representations across the timestamp axis
                aggregated = encoded_blocks.sum(dim=1)

                # Append the summed representations to the first k raw values
                first_k = x[:, :self.k].to(aggregated.dtype)  # Ensure same dtype of entries of x and representations

                return torch.cat([first_k, aggregated], dim=1)

            def forward(self, x):
                x = self._apply_encoder(x)
                x = nn.functional.relu(self.head_in(x))
                x = nn.functional.relu(self.head_hidden1(x))
                x = nn.functional.relu(self.head_hidden2(x))
                x = self.head_out(x)
                return x


        # ----- PEN -------
        print("Creating PEN:")
        
        training_trajs = self._simulate_training_trajs(training_thetas, traj_initial_value, model_simulator, timestep, num_samples=len(real_trajectory))

        #  Ensure correct type for training data
        training_trajs = np.asarray(training_trajs, dtype=np.float32)
        training_thetas = np.asarray(training_thetas, dtype=np.float32)

        # Validate shapes for training data
        if training_trajs.ndim != 2:
            raise ValueError("training_data_x must be a 2D array of shape (num_samples, sequence_length)")
        if training_thetas.ndim != 2:
            raise ValueError("training_data_params must be a 2D array of shape (num_samples, parameter_dim)")
        if training_trajs.shape[0] != training_thetas.shape[0]:
            raise ValueError("training_data_x and training_data_params must have the same number of samples")

        # Create instance of PEN (torch neural network)
        summaryNN = PEN(parameter_dim = training_thetas[0].size, markov_order=markov_order)

        # Validate user-input training device
        allowed = {"cpu", "cuda"}
        if device_name is None:
            print("No device specified: training device set to CPU")
            device_name = "cpu"
        elif device_name not in allowed:
            raise ValueError(f"Invalid device '{device_name}'. Must be one of {allowed}.")

        # Move summaryNN to training device
        if device_name == "cpu":
            print("Training device set to CPU")
            device = torch.device("cpu")
        elif device_name == "cuda" and torch.cuda.is_available():
            print("Training device set to GPU")
            device = torch.device("cuda") 
        else:
            print("GPU not available: training device set to CPU")
            device = torch.device("cpu")
        summaryNN.to(device)
 
        # Prepare data-loader for training
        dataset = TensorDataset(
            torch.from_numpy(training_trajs),
            torch.from_numpy(training_thetas),
        )

        # Train/val split for early stopping (always enabled; falls back to train loss if no split)
        total_n = len(dataset)
        bs = min(batch_size, total_n) if total_n > 0 else batch_size
        if validation_split > 0.0 and total_n > 1:
            val_n = max(1, int(total_n * float(validation_split)))
            train_n = max(1, total_n - val_n)
            train_subset, val_subset = torch.utils.data.random_split(dataset, [train_n, val_n])
            loader = DataLoader(train_subset, batch_size=bs, shuffle=True)
            val_loader = DataLoader(val_subset, batch_size=bs, shuffle=False)
        else:
            loader = DataLoader(dataset, batch_size=bs, shuffle=True)
            val_loader = None
        
        # Specify cost function
        weights = torch.from_numpy(dimension_weights**2).to(device)
        criterion = lambda pred, target: (F.mse_loss(pred, target, reduction="none")*weights).mean()

        # Specify optimiser for training
        optimizer = torch.optim.Adam(summaryNN.parameters(), lr=1e-3)

        # Enter training
        patience = max(1, int(early_stopping_patience))
        print(
            f"Beginning training loop... | early stopping: patience={patience}, "
            f"min_delta={float(early_stopping_loss_drop):.2f}, validation_split={float(validation_split):.2f}"
        )
        summaryNN.train()
        best_loss = float("inf")
        no_improve = 0
        best_state = None  # save best-performing model weights
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            n_batches = 0
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)

                preds = summaryNN(batch_x)
                loss = criterion(preds, batch_y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += float(loss.detach().cpu().item())
                n_batches += 1

            # Compute average training loss for epoch
            train_avg = epoch_loss / max(1, n_batches)
            current = train_avg
            if val_loader is not None:
                summaryNN.eval()
                with torch.no_grad():
                    val_loss = 0.0
                    val_batches = 0
                    for vx, vy in val_loader:
                        vx = vx.to(device)
                        vy = vy.to(device)
                        vp = summaryNN(vx)
                        vloss = criterion(vp, vy)
                        val_loss += float(vloss.detach().cpu().item())
                        val_batches += 1
                val_avg = val_loss / max(1, val_batches)
                current = val_avg
                summaryNN.train()
            else:
                val_avg = None

            # Per-epoch logging
            if val_avg is None:
                print(f"Epoch {epoch+1:3d}/{num_epochs} - train_loss: {train_avg:.6f} - best: {best_loss:.6f} - patience: {no_improve}/{patience}")
            else:
                print(f"Epoch {epoch+1:3d}/{num_epochs} - train_loss: {train_avg:.6f} - val_loss: {val_avg:.6f} - best: {best_loss:.6f} - patience: {no_improve}/{patience}")

            # Early stopping check (always on)
            if current < best_loss - float(early_stopping_loss_drop):
                print(f"  Improvement: best {best_loss:.6f} -> {current:.6f}")
                best_loss = current
                no_improve = 0
                # Save the best model state whenever best improves
                best_state = {k: v.detach().cpu().clone() for k, v in summaryNN.state_dict().items()}
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"Early stopping at epoch {epoch+1}: best_loss={best_loss:.6f}, current={current:.6f}")
                    # Restore best model weights before exiting training loop
                    if best_state is not None:
                        summaryNN.load_state_dict(best_state)
                    break
        summaryNN.eval()
        print("finished!")
        # Exit training

        # ----- Export to lightweight numpy version -----

        def grab(layer):
            W = layer.weight.detach().cpu().numpy().astype(np.float32)
            b = layer.bias.detach().cpu().numpy().astype(np.float32)
            return W, b
        
        layers = {
            "encoder_in":     grab(summaryNN.encoder_in),
            "encoder_hidden": grab(summaryNN.encoder_hidden),
            "encoder_out":    grab(summaryNN.encoder_out),
            "head_in":        grab(summaryNN.head_in),
            "head_hidden1":   grab(summaryNN.head_hidden1),
            "head_hidden2":   grab(summaryNN.head_hidden2),
            "head_out":       grab(summaryNN.head_out),
        }

        self.layers = layers
        self.k = markov_order

        # ------ Call base constructor now that we can summarise ------
        super().__init__(real_trajectory, timestep)

    # ----- Save/load trained PEN weights -----
    def save_pen(self, filepath: str):
        """Save trained PEN weights and markov order to a .npz file.

        Args:
            filepath: Destination path (e.g., "pen_weights.npz").
        """
        if self.layers is None or self.k is None:
            raise RuntimeError("PEN not initialised. Train or load it before saving.")

        # Flatten the layer dict for easy storage in NPZ
        npz_dict = {"k": np.asarray(self.k, dtype=np.int32)}
        for name, (W, b) in self.layers.items():
            npz_dict[f"{name}_W"] = np.asarray(W, dtype=np.float32)
            npz_dict[f"{name}_b"] = np.asarray(b, dtype=np.float32)

        np.savez_compressed(filepath, **npz_dict)

    @classmethod
    def load_pen(cls, filepath: str, real_trajectory, timestep):
        """Create a CalculatePENDistance instance from a saved .npz file.

        Loads the trained PEN weights and configures the instance so it can be
        used immediately as a distance calculator (summary of real_trajectory
        is computed on construction).

        Args:
            filepath: Path to the saved .npz produced by save_pen.
            real_trajectory: The reference trajectory to summarise.
            timestep: Sampling interval of the trajectory.

        Returns:
            CalculatePENDistance: Configured instance ready for eval().
        """
        data = np.load(filepath)
        # Reconstruct layers dict
        expected = [
            "encoder_in", "encoder_hidden", "encoder_out",
            "head_in", "head_hidden1", "head_hidden2", "head_out",
        ]

        layers = {}
        for name in expected:
            W_key = f"{name}_W"
            b_key = f"{name}_b"
            if W_key not in data or b_key not in data:
                raise ValueError(f"Missing keys '{W_key}'/'{b_key}' in saved PEN file")
            layers[name] = (data[W_key].astype(np.float32), data[b_key].astype(np.float32))

        if "k" not in data:
            raise ValueError("Missing key 'k' (markov order) in saved PEN file")
        k = int(np.asarray(data["k"]).item())

        # Build and initialise instance
        obj = cls()
        obj.layers = layers
        obj.k = k
        # Call base initialiser to compute real-data summary using loaded PEN
        CalculateDistance.__init__(obj, real_trajectory, timestep)
        return obj


    # ------ Execute and return feed foward ------
    def _summarise(self, trajectory):
        # Helpers for feeding forward
        def linear(z, W, b):
            return z@W.T + b
        def ReLU(z):
            return np.maximum(z, 0.0)

        trajectory = np.asarray(trajectory, dtype=np.float32)

        if self.k is None:
            raise RuntimeError("PEN not initialised. Call create_and_train_PEN first.")
        if trajectory.size < self.k + 1:
            raise ValueError(f"Input length ({trajectory.size}) must be at least markov_order+1 ({self.k + 1}).")

        # Numba-accelerated feed-forward
        W_e_in, b_e_in = self.layers["encoder_in"]
        W_e_h, b_e_h = self.layers["encoder_hidden"]
        W_e_out, b_e_out = self.layers["encoder_out"]
        W_h_in, b_h_in = self.layers["head_in"]
        W_h1, b_h1 = self.layers["head_hidden1"]
        W_h2, b_h2 = self.layers["head_hidden2"]
        W_h_out, b_h_out = self.layers["head_out"]

        return _pen_feedforward_jit(
            trajectory, int(self.k),
            W_e_in, b_e_in,
            W_e_h, b_e_h,
            W_e_out, b_e_out,
            W_h_in, b_h_in,
            W_h1, b_h1,
            W_h2, b_h2,
            W_h_out, b_h_out,
        )
        return x


    def _calculate_summaries_distance(self, simulation_summary):
        return np.linalg.norm(self.summary-simulation_summary)
