from abc import ABC, abstractmethod
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from KDEpy.FFTKDE import FFTKDE
from scipy import signal, integrate

import matplotlib.pyplot as plt




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
        
        # FOR TESTING PURPOSES
        # plt.plot(grid, sim_pdf, linestyle="dotted")
        # plt.plot(grid, real_pdf)
        # plt.show()
        # plt.plot(freqs, spectral_density2, linestyle="dotted")
        # plt.plot(freqs, spectral_density1)
        # plt.show()

        # Compute integrated absolute differences, combine via IAE1 + alpha*IAE2
        pdf_distance = integrate.trapezoid(np.abs(real_pdf - sim_pdf), grid)
        spectral_density_distance = integrate.trapezoid(y = np.abs(spectral_density1 - spectral_density2), x = freqs)
        alpha = integrate.trapezoid(y = np.abs(spectral_density1), x = freqs)

        return spectral_density_distance + alpha*pdf_distance
    





    
class CalculatePENDistance(CalculateDistance):
    def __init__(
        self
    ):
        self.layers = None  # Dictionary containing PEN weights and biases
        self.k = None       # Markov order

    # ----- Alternative Constructor -----
    def create_and_train_PEN(
            self, 
            model_simulator: callable, training_thetas, traj_initial_value, real_trajectory, timestep, 
            num_epochs, 
            markov_order = 1, 
            device_name = "cpu",
            batch_size=32,
            early_stopping_patience=10,
            early_stopping_min_delta=0.0,
            validation_split=0.1,
            ):
        
        # Lazy imports to prevent torch dependency in instances of this class
        import torch
        import torch.nn as nn
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

        # Create training data from simulator
        print("Creating PEN:")
        print("Simulating training trajectories...")
        training_trajs = []
        for theta in training_thetas:
            training_trajs.append(model_simulator(traj_initial_value, theta, timestep, real_trajectory.size))
        training_trajs = np.array(training_trajs)
        print("finished!")
        
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
        
        # Define cost function
        def mean_euclidean_squared(pred, target):
            sample_loss = (pred - target).pow(2).sum(dim=1)
            return sample_loss.mean()
        
        # Specify optimiser for training
        optimizer = torch.optim.Adam(summaryNN.parameters(), lr=1e-3)

        # Enter training
        patience = max(1, int(early_stopping_patience))
        print(
            f"Beginning training loop... | early stopping: patience={patience}, "
            f"min_delta={float(early_stopping_min_delta):.2e}, validation_split={float(validation_split):.2f}"
        )
        summaryNN.train()
        best_loss = float("inf")
        no_improve = 0
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            n_batches = 0
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)

                preds = summaryNN(batch_x)
                loss = mean_euclidean_squared(preds, batch_y)

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
                        vloss = mean_euclidean_squared(vp, vy)
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
            if current < best_loss - float(early_stopping_min_delta):
                print(f"  Improvement: best {best_loss:.6f} -> {current:.6f}")
                best_loss = current
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"Early stopping at epoch {epoch+1}: best_loss={best_loss:.6f}, current={current:.6f}")
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

        # Create contiguous blocks for encoder
        blocks = sliding_window_view(trajectory, window_shape = self.k+1)

        # Extract weights and biases 
        W, b = self.layers["encoder_in"]
        blocks = ReLU(linear(blocks, W, b))
        W, b = self.layers["encoder_hidden"]
        blocks = ReLU(linear(blocks, W, b))
        W, b = self.layers["encoder_out"]
        blocks = ReLU(linear(blocks, W, b))

        encoding = blocks.sum(axis=0)
        first_k = trajectory[:self.k]
        x = np.concatenate([first_k, encoding], axis=0)

        W, b = self.layers["head_in"];       x = ReLU(linear(x, W, b)); 
        W, b = self.layers["head_hidden1"];  x = ReLU(linear(x, W, b)); 
        W, b = self.layers["head_hidden2"];  x = ReLU(linear(x, W, b)); 
        W, b = self.layers["head_out"];      x = linear(x, W, b); 

        return x


    def _calculate_summaries_distance(self, simulation_summary):
        return np.linalg.norm(self.summary-simulation_summary)
