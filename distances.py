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
        grid = np.linspace(lb, ub, 1000) # grid for sampmles for estimated density

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
        encoded = nn.functional.relu(self.encoder_in(blocks))
        encoded = nn.functional.relu(self.encoder_hidden(encoded))
        encoded = nn.functional.relu(self.encoder_out(encoded))
        encoded = encoded.reshape(batch_size, num_blocks, encoded.size(-1))

        # Sum the representations across the timestamp axis
        aggregated = encoded.sum(dim=1)

        # Append the summed representations to the first k raw values
        first_k = x[:, :self.k].to(aggregated.dtype)  # Ensure same dtype of entries of x and representations
        encoded_sequence = torch.cat([first_k, aggregated], dim=1)

        return encoded_sequence

    def forward(self, x):
        '''Forward pass of PEN network. Input is a batch of time series.'''

        x = self._apply_encoder(x)
        x = nn.functional.relu(self.head_in(x))
        x = nn.functional.relu(self.head_hidden1(x))
        x = nn.functional.relu(self.head_hidden2(x))
        x = self.head_out(x)

        return x



class CalculatePENDistance(CalculateDistance):
    def __init__(self, real_trajectory, timestep, parameter_dim):
        super().__init__(real_trajectory, timestep)
        self.summary = PEN(parameter_dim)

    def _train_model(self, model_simulator: callable, batch_size=32, n_epochs=1000):
        batches = list([i,min(i+batch_size-1,n_epochs)] for i in range(0,n_epochs,batch_size))
        data = 