from abc import ABC, abstractmethod
import numpy as np
from scipy import stats, signal, integrate
import concurrent.futures
import tqdm
import torch
import torch.nn as nn


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

    def compare_trajectory(self, simulation_trajectory):
        summary = self._summarise(simulation_trajectory)
        return self._calculate_summaries_distance(summary)



class CalculateModelBasedDistance(CalculateDistance):
    """Calculate trajectory distance using estimated density and spectral density as summaries."""
    def _summarise(self, trajectory):
        frequencies, spectral_density = signal.periodogram(trajectory, self.timestep)
        kde = stats.gaussian_kde(trajectory)
        kde_support = np.linspace(trajectory.min(), trajectory.max(), 1000)  # fixed grid
        kde_values = kde.evaluate(kde_support)
        return (kde_support, kde_values, frequencies, spectral_density)

    def _calculate_summaries_distance(self, simulation_summary):
        support2, kde_values2, frequencies2, spectral_density2 = simulation_summary
        support1, kde_values1, frequencies1, spectral_density1 = self.summary
        if (frequencies1 != frequencies2).all():
            raise ValueError("Periodogram frequencies do not match. Ensure that the time duration of both trajectories is the same.")

        lb = min(support1[0], support2[0])
        ub = max(support1[1], support2[1])
        grid = np.linspace(lb, ub, 1000) # Grid for sampmles for estimated density

        kde1_interp = np.interp(grid, support1, kde_values1)
        kde2_interp = np.interp(grid, support2, kde_values2)

        pdf_distance = integrate.trapezoid(np.abs(kde1_interp - kde2_interp), grid)
        spectral_density_distance = integrate.trapezoid(y = np.abs(spectral_density1 - spectral_density2), x = frequencies1)
        alpha = np.abs(integrate.trapezoid(y = spectral_density1, x = frequencies1))
        
        return spectral_density_distance + alpha*pdf_distance
    


class PEN(nn.Module):
    '''Partially Exchangeable Network for learning summary statistics for Markov process'''
    def __init__(self, parameter_dim, markov_order=1):
        super().__init__()
        self.k = markov_order

        self.encoder_in = nn.Linear(self.k+1, 100)
        self.encoder_hidden = nn.Linear(100,50)
        self.encoder_out = nn.Linear(50,10)

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
        '''Forward pass of PEN network. Input is a batch of time series.'''

        x = self._apply_encoder(x)
        x = nn.functional.relu(self.head_in(x))
        x = nn.functional.relu(self.head_hidden1(x))
        x = nn.functional.relu(self.head_hidden2(x))
        x = self.head_out(x)

        return x



class CalculatePENDistance(CalculateDistance):
    def __init__(self, real_trajectory, timestep, parameter_dim, training_data_x, training_data_params, batch_size=32, device=None):
        super().__init__(real_trajectory, timestep)

        self.summaryNN = PEN(parameter_dim)
        if device is not None:
            self.summaryNN.to(device)

        self._train_model(training_data_x=training_data_x, training_data_params=training_data_params, batch_size=batch_size)
        with torch.no_grad():
            x = torch.from_numpy(self.real_trajectory).float().unsqueeze(0).todevice(device)
            self.summary = self.summaryNN(x).squeeze(0) # 


    def _train_model(self, training_data_x, training_data_params, batch_size=32, num_epochs=10):
        training_array = np.asarray(training_data_x)
        param_array = np.asarray(training_data_params)
        if training_array.ndim != 1:
            raise ValueError("training_data must be a 1D numpy array or vector")
        if param_array.ndim != 2:
            raise ValueError("training_data_params must be a 2D numpy array")

        # Ensure batch size doesn't exceed sample size, and ensure training data matches
        num_samples = training_array.shape[0]
        if num_samples < batch_size:
            batch_size = num_samples
        if param_array.shape[0] != num_samples:
            raise ValueError(
                "training_data_params must have the same number of rows as training_data_x has samples"
            )

        # Create (first, last) index pairs for each training batch
        batches = [
            (first, min(first + batch_size, num_samples))
            for first in range(0, num_samples, batch_size)
        ]

        tensor_batches = []
        tensor_param_batches = []
        for first, last in batches:
            batch_np = training_array[first:last]
            batch_tensor = torch.from_numpy(batch_np.copy()).float().unsqueeze(0)
            tensor_batches.append(batch_tensor)

            param_np = param_array[first:last]
            param_tensor = torch.from_numpy(param_np.copy()).float().unsqueeze(0)
            tensor_param_batches.append(param_tensor)

        def batch_mean_squared_euclidean(pred, target):
            sample_loss = (pred - target).pow(2).sum(dim=1)
            return sample_loss.mean()

        # Find device and set up optimiser
        device = next(self.summaryNN.parameters()).device
        optimizer = torch.optim.Adam(self.summaryNN.parameters(), lr=1e-3)

        # Training loop
        self.summaryNN.train()
        for _ in range(num_epochs):
            for input_batch, param_batch in zip(tensor_batches, tensor_param_batches):
                input_batch = input_batch.to(device)
                param_batch = param_batch.to(device)
                target = param_batch.squeeze(0)  # shape (batch_len, param_dim)
                prediction = self.summaryNN(input_batch).squeeze(0)
                loss = batch_mean_squared_euclidean(prediction, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        # Switch back to evaluation mode
        self.summaryNN.eval()


    def _calculate_summaries_distance(self, simulation_summary):
        # Euclidean distance between summary statistics
        diff = self.summary - simulation_summary
        return torch.linalg.norm(diff, ord=2).item()  # Return as float