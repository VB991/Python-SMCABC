from abc import ABC, abstractmethod
import numpy as np
from scipy import stats, signal, integrate
from torch import nn
import matplotlib.pyplot as plt


class CalculateDistance(ABC):
    """Abstract base class for trajectory distance calculators."""
    @abstractmethod
    def _summarise(trajectory):
        pass

    @abstractmethod
    def _distance_from_summary(self, simulation_summary):
        pass

    def __init__(self, real_trajectory, timestep):
        self.timestep = timestep
        self.summary = self._summarise(real_trajectory)

    def compare_trajectory(self, simulation_trajectory):
        summary = self._summarise(simulation_trajectory)
        return self._distance_from_summary (summary)


class CalculateModelBasedDistance(CalculateDistance):
    """Calculate trajectory distance using estimated density and spectral density as summaries."""
    def _summarise(self, trajectory):
        kde_pdf = stats.gaussian_kde(trajectory).pdf
        frequencies, spectral_density = signal.periodogram(trajectory, self.timestep)
        kde_support = [trajectory.min(),trajectory.max()]
        return (kde_support, kde_pdf, frequencies, spectral_density)

    def _distance_from_summary(self, simulation_summary):
        support2, kde_pdf2, frequencies2, spectral_density2 = simulation_summary
        support1, kde_pdf1, frequencies1, spectral_density1 = self.summary
        if (frequencies1 != frequencies2).all():
            raise ValueError("Periodogram frequencies do not match. Ensure that the time duration of both trajectories is the same.")
        
        lb = min(support1[0],support2[0])
        ub = max(support1[1],support2[1])
        
        pdf_distance = integrate.quad(lambda x: np.abs(kde_pdf1(x) - kde_pdf2(x)), lb, ub)[0]
        spectral_density_distance = integrate.trapezoid(y = np.abs(spectral_density1 - spectral_density2), x = frequencies1)
        alpha = np.abs(integrate.trapezoid(y = spectral_density1, x = frequencies1))
        
        return spectral_density_distance + alpha*pdf_distance
    

class PEN(nn.Module):

    def __init__(self):
        super().__init__()

        self.outer_input2