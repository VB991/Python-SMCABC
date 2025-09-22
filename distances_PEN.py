from distances import CalculateDistance
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

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
    def __init__(
        self,
        real_trajectory,
        timestep,
        parameter_dim,
        training_data_x,
        training_data_params,
        markov_order=1,
        batch_size=32,
        num_epochs=10,
        device=None,
    ):
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.summaryNN = PEN(parameter_dim=parameter_dim, markov_order=markov_order).to(self.device)

        self._train_model(
            training_data_x=training_data_x,
            training_data_params=training_data_params,
            batch_size=batch_size,
            num_epochs=num_epochs,
        )

        super().__init__(real_trajectory, timestep)



    def _train_model(self, training_data_x, training_data_params, batch_size=32, num_epochs=10):
        training_array = np.asarray(training_data_x, dtype=np.float32)
        param_array = np.asarray(training_data_params, dtype=np.float32)

        if training_array.ndim != 2:
            raise ValueError("training_data_x must be a 2D array of shape (num_samples, sequence_length)")
        if param_array.ndim != 2:
            raise ValueError("training_data_params must be a 2D array of shape (num_samples, parameter_dim)")
        if training_array.shape[0] != param_array.shape[0]:
            raise ValueError("training_data_x and training_data_params must have the same number of samples")

        dataset = TensorDataset(
            torch.from_numpy(training_array),
            torch.from_numpy(param_array),
        )
        loader = DataLoader(dataset, batch_size=min(batch_size, len(dataset)), shuffle=True)

        def mean_euclidean_squared(pred, target):
            sample_loss = (pred - target).pow(2).sum(dim=1)
            return sample_loss.mean()

        optimizer = torch.optim.Adam(self.summaryNN.parameters(), lr=1e-3)

        # Enter training
        self.summaryNN.train()
        for _ in range(num_epochs):
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                preds = self.summaryNN(batch_x)
                loss = mean_euclidean_squared(preds, batch_y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        # Exit training mode
        self.summaryNN.eval()


    def _summarise(self, trajectory):
        traj_tensor = torch.as_tensor(trajectory, dtype=torch.float32)
        if traj_tensor.dim() == 1:
            traj_tensor = traj_tensor.unsqueeze(0)
        elif traj_tensor.dim() != 2:
            raise ValueError("trajectory must be 1D or 2D with shape (batch, sequence_length)")

        traj_tensor = traj_tensor.to(self.device)
        with torch.no_grad():
            summary = self.summaryNN(traj_tensor).squeeze(0).cpu()
        return summary


    def _calculate_summaries_distance(self, simulation_summary):
        diff = self.summary - simulation_summary
        if not isinstance(diff, torch.Tensor):
            diff = torch.as_tensor(diff, dtype=torch.float32)
        return torch.linalg.norm(diff.float(), ord=2).item()
