"""Model of the forecast error, used as noise in the Monte-Carlo simulation.

The reported log incidence y_t is the true log incidence x_t plus counting noise, and the true
log incidence follows the forecasting model up to a persistent process error:

    y_t = x_t + nu_t,                          nu_t ~ N(0, tau2 / n_t)
    x_(t+1) = x_t + f(...) + eta_(t+1),        eta_(t+1) = phi * eta_t + xi_(t+1)

n_t are the weekly cases (Poisson counting noise gives tau2 = 1) and xi ~ N(0, sigma_eta^2 (1 - phi^2)),
so that eta has the stationary standard deviation sigma_eta. The residual of an observed weekly
change, e_(t+1) = eta_(t+1) + nu_(t+1) - nu_t, then has the moments

    E[e_(t+1)^2]       = sigma_eta^2 + 2 tau2 / n_t
    E[e_(t+1) e_t]     = phi sigma_eta^2 - tau2 / n_t

The counting noise makes consecutive residuals negatively correlated in weeks with few cases, the
persistent process error positively correlated in large epidemics; the moments identify the three
parameters from out-of-fold residuals.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from utils.epi_calendar import ONE_WEEK

# Reporting dips around Christmas and New Year are transient: they are left out of the fit
# instead of being propagated as noise of the epidemic dynamics
TURN_OF_YEAR_WEEKS = (51, 52, 53)
# Lower bound of the case count, limits the counting noise in weeks with (almost) no cases
MIN_CASES = 1.0


def residual_pairs(residuals: pd.DataFrame) -> pd.DataFrame:
    """Residuals of consecutive weeks of the same city, outside the turn of the year.

    Args:
        residuals: One row per week with columns city_name, date, epi_week, cases and residual.

    Returns:
        DataFrame with columns residual (week t), previous (week t-1) and cases (week t).
    """
    table = residuals.sort_values(["city_name", "date"])
    by_city = table.groupby("city_name")
    consecutive = (
        by_city["date"].diff().eq(ONE_WEEK)
        & ~table["epi_week"].isin(TURN_OF_YEAR_WEEKS)
        & ~by_city["epi_week"].shift(1).isin(TURN_OF_YEAR_WEEKS)
    )
    return pd.DataFrame({
        "residual": table["residual"],
        "previous": by_city["residual"].shift(1),
        "cases": table["cases"],
    })[consecutive]


@dataclass(frozen=True)
class ErrorModel:
    """Persistent process error (sigma_eta, phi) and transient counting noise (tau2), see module docstring."""

    sigma_eta: float
    phi: float
    tau2: float

    @classmethod
    def fit(cls, residuals: pd.DataFrame) -> "ErrorModel":
        """Fit the parameters to out-of-fold residuals by the method of moments.

        The two moment equations are linear in (sigma_eta^2, tau2, phi * sigma_eta^2) and are
        solved together by least squares; phi is restricted to [0, 0.99].

        Args:
            residuals: One row per week with columns city_name, date, epi_week, cases (weekly
                cases in week t) and residual (observed minus predicted change from t to t+1).
        """
        regular = residuals[~residuals["epi_week"].isin(TURN_OF_YEAR_WEEKS)]
        pairs = residual_pairs(residuals)
        n_regular = np.maximum(regular["cases"].to_numpy(), MIN_CASES)
        n_pairs = np.maximum(pairs["cases"].to_numpy(), MIN_CASES)

        design = np.vstack([
            np.column_stack([np.ones_like(n_regular), 2 / n_regular, np.zeros_like(n_regular)]),
            np.column_stack([np.zeros_like(n_pairs), -1 / n_pairs, np.ones_like(n_pairs)]),
        ])
        moments = np.concatenate([
            regular["residual"].to_numpy() ** 2,
            (pairs["residual"] * pairs["previous"]).to_numpy(),
        ])
        (variance_eta, tau2, covariance_eta), *_ = np.linalg.lstsq(design, moments, rcond=None)
        if variance_eta <= 0 or tau2 < 0:
            raise ValueError(f"Invalid error model: sigma_eta^2={variance_eta:.4f}, tau2={tau2:.4f}")

        phi = float(np.clip(covariance_eta / variance_eta, 0.0, 0.99))
        return cls(sigma_eta=float(np.sqrt(variance_eta)), phi=phi, tau2=float(tau2))

    def counting_noise(self, cases: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Counting noise nu of the reported log incidence for the given weekly cases."""
        return rng.normal(0.0, np.sqrt(self.tau2 / np.maximum(cases, MIN_CASES)))

    def initial_process_error(self, size: int, rng: np.random.Generator) -> np.ndarray:
        """Process error eta at the start, drawn from its stationary distribution."""
        return rng.normal(0.0, self.sigma_eta, size)

    def next_process_error(self, eta: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Process error of the next week, eta_(t+1) = phi * eta_t + xi_(t+1)."""
        return self.phi * eta + rng.normal(0.0, self.sigma_eta * np.sqrt(1 - self.phi**2), len(eta))

    def change_std(self, cases) -> np.ndarray:
        """Standard deviation of the residual of an observed weekly change."""
        return np.sqrt(self.sigma_eta**2 + 2 * self.tau2 / np.maximum(cases, MIN_CASES))

    def change_autocorrelation(self, cases) -> np.ndarray:
        """Correlation of the residuals of two consecutive weekly changes."""
        n = np.maximum(cases, MIN_CASES)
        return (self.phi * self.sigma_eta**2 - self.tau2 / n) / (self.sigma_eta**2 + 2 * self.tau2 / n)

    def to_dict(self) -> dict:
        return {"sigma_eta": self.sigma_eta, "phi": self.phi, "tau2": self.tau2}

    @classmethod
    def from_dict(cls, values: dict) -> "ErrorModel":
        return cls(sigma_eta=values["sigma_eta"], phi=values["phi"], tau2=values["tau2"])
