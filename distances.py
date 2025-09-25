from abc import ABC, abstractmethod
import numpy as np
from KDEpy.FFTKDE import FFTKDE
from scipy import signal, integrate, fft

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

        # Extract components of the "real" summary, but do NOT keep the KDE object
        ends, _kde_unused, *rest = self.summary
        frequencies, smooth_spectral_density = rest

        # Build a fixed grid of the real-data support, and evaluate pdf_real
        self.grid = np.linspace(ends[0], ends[1], 1024)
        self.grid_spacing = float(self.grid[1] - self.grid[0])
        kde_real = FFTKDE(kernel="gaussian", bw="silverman")
        kde_real.fit(real_trajectory)
        self.pdf = kde_real.evaluate(self.grid)

        # Omit unpicklable kde object (for multiprocessing)
        self.summary = (ends, None, frequencies, smooth_spectral_density)

    def _summarise(self, trajectory):       
        # KDE object for estimated density, support for KDE
        kde = FFTKDE(kernel="gaussian", bw="silverman")
        kde.fit(trajectory)
        padding = 2*trajectory.std()    # ensure bulk of pdf is contained
        kde_support_ends = (trajectory.min()-padding, trajectory.max()+padding) 

        # Spectral density (similarly to R's "spectrum")
        # Create modified boxcar kernel
        ker = np.ones(self.span, float)
        ker[0] = ker[-1] = 0.5
        ker /= (self.span-1)
        # Compute periodogram and smooth with kernel (with wraparound padding)
        frequencies, spectral_density = signal.periodogram(
            x = trajectory,
            nfft = fft.next_fast_len(len(trajectory)),
            fs = 1/self.timestep,
            window = ("tukey", 0.2),
            return_onesided=True, 
            detrend = "linear",
            scaling="density"
            )
        # Pad with wraparound padding to allow convolution at endpoints
        pad_length = int((self.span-1)/2)
        padded_density = np.pad(spectral_density, pad_length, mode="wrap")
        smooth_spectral_density = signal.fftconvolve(padded_density, ker, mode="valid")

        # Endpoints for pdf support, kde object, frequencies, and spectral density at frequencies
        return (kde_support_ends, kde, frequencies, smooth_spectral_density)

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
