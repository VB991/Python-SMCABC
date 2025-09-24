from abc import ABC, abstractmethod
import numpy as np
from KDEpy.FFTKDE import FFTKDE
from scipy import signal, integrate

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
    def __init__(self, real_trajectory, timestep, span:int = 51):
        """Calculate trajectory distance using estimated density and spectral density as summaries.

        Args:
            real_trajectory (_type_): Real data
            timestep (_type_): Timestep of data
            span (_type_): Span used for modified boxcar kernel when smoothing periodogram. Defaults to 51.
        """
        # Validate span for periodogram smoothing kernel
        if span < 3 or span%2 == 0:
            raise ValueError("Span of periodogram smoothing kernel must be an odd integer, at least 3")
        self.span = span

        super().__init__(real_trajectory, timestep)

        ends, kde, *_ = self.summary
        self.grid = np.linspace(ends[0], ends[1], 1024)
        self.grid_spacing = self.grid[1] - self.grid[0]
        self.pdf = kde.evaluate(self.grid)

    def _summarise(self, trajectory):       
        # KDE object for estimated density, support for KDE
        kde = FFTKDE(kernel="gaussian", bw="silverman")
        kde.fit(trajectory)
        padding = 2*trajectory.std()    # ensure bulk of pdf is contained
        kde_support_ends = (trajectory.min()-padding, trajectory.max()+padding) 

        # Spectral density (similarly to R's "spectrum")
        # Create modified boxcar kernel, remove linear trend from data
        ker = np.ones(self.span, float)
        ker[0] = ker[-1] = 0.5
        ker /= (self.span-1)
        detr_traj = signal.detrend(trajectory, type="linear")
        # Compute periodogram and smooth with kernel (with wraparound padding)
        frequencies, spectral_density = signal.periodogram(detr_traj, 1/self.timestep, window=("tukey", 0.2),return_onesided=True, scaling="density")
        pad_length = int((self.span-1)/2)
        padded_density = np.pad(spectral_density, pad_length, mode="wrap")
        smooth_spectral_density = np.convolve(padded_density, ker, mode="valid")

        # Endpoints for pdf support, kde object, frequencies, and spectral density at frequencies
        return (kde_support_ends, kde, frequencies, smooth_spectral_density)

    def _calculate_summaries_distance(self, simulation_summary):
        ends1, kde1, frequencies1, spectral_density1 = self.summary
        ends2, kde2, frequencies2, spectral_density2 = simulation_summary

        if not np.allclose(frequencies1, frequencies2):
            raise ValueError("Periodogram frequencies do not match. Ensure that the time duration of both trajectories is the same.")
        else:
            freqs = frequencies1

        # Evaluate kde over same grid
        left_diff = ends1[0] - ends2[0]
        right_diff = ends2[1] - ends1[1] 
        n_left_points = 0
        n_right_points = 0
        extra_left_points = np.empty(0)
        extra_right_points = np.empty(0)
        # Extend grid left
        if left_diff > 0:
            n_left_points = -(-left_diff // self.grid_spacing)  # round up mod division
            extra_left_points = np.arange(n_left_points)*self.grid_spacing + ends2[0]
        if right_diff > 0:
            n_right_points = -(-right_diff // self.grid_spacing) 
            extra_right_points = np.arange(n_right_points)*self.grid_spacing + ends1[1] + self.grid_spacing
        grid = np.concatenate((extra_left_points, self.grid, extra_right_points))

        sim_pdf = kde2.evaluate(grid)

        # Compute integrated absolute differences, combine via IAE1 + alpha*IAE2
        pdf_distance = integrate.trapezoid(np.abs(self.pdf - sim_pdf), grid)
        spectral_density_distance = integrate.trapezoid(y = np.abs(spectral_density1 - spectral_density2), x = freqs)
        alpha = integrate.trapezoid(y = np.abs(spectral_density1), x = freqs)

        return spectral_density_distance + alpha*pdf_distance



