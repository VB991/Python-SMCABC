from abc import ABC, abstractmethod
import numpy as np
from KDEpy.FFTKDE import FFTKDE
from scipy import signal, integrate, fft

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
            spans = [n]
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
        window = signal.windows.tukey(M=trajectory.size, alpha=0.1)
        frequencies, spectral_density = signal.periodogram(
            trajectory,
            fs=fs,
            window=window,
            detrend="constant",
            return_onesided=True,
            scaling="density",
        )

        # Smooth with modified boxcar kernels (ends half the weight)
        spectral_density = signal.fftconvolve(spectral_density, self.smooth_ker, mode="same")

        return frequencies, spectral_density

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
