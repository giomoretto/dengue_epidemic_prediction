import numpy as np
import pandas as pd
import pytest

from model_definition.error_model import ErrorModel, residual_pairs

SIGMA_ETA, PHI, TAU2 = 0.2, 0.4, 1.0


@pytest.fixture
def synthetic_residuals() -> pd.DataFrame:
    """Residuals e_(t+1) = eta_(t+1) + nu_(t+1) - nu_t simulated with known parameters, for cities
    with 5 to 10 000 weekly cases."""
    rng = np.random.default_rng(0)
    frames = []
    for cases in (5, 20, 100, 1_000, 10_000):
        n_weeks = 4_000
        eta = np.empty(n_weeks)
        eta[0] = rng.normal(0, SIGMA_ETA)
        for t in range(1, n_weeks):
            eta[t] = PHI * eta[t - 1] + rng.normal(0, SIGMA_ETA * np.sqrt(1 - PHI**2))
        nu = rng.normal(0, np.sqrt(TAU2 / cases), n_weeks + 1)
        frames.append(pd.DataFrame({
            "city_name": f"city with {cases} cases",
            "date": pd.date_range("1950-01-01", periods=n_weeks, freq="7D"),
            "epi_week": 10,
            "cases": float(cases),
            "residual": eta + nu[1:] - nu[:-1],
        }))
    return pd.concat(frames, ignore_index=True)


def test_fit_recovers_the_parameters(synthetic_residuals):
    error_model = ErrorModel.fit(synthetic_residuals)

    assert error_model.sigma_eta == pytest.approx(SIGMA_ETA, rel=0.1)
    assert error_model.phi == pytest.approx(PHI, abs=0.07)
    assert error_model.tau2 == pytest.approx(TAU2, rel=0.1)


def test_turn_of_year_weeks_are_left_out_of_the_fit(synthetic_residuals):
    holiday = synthetic_residuals.head(500).assign(city_name="holiday city", epi_week=52, residual=3.0)

    with_holidays = ErrorModel.fit(pd.concat([synthetic_residuals, holiday], ignore_index=True))

    assert with_holidays == ErrorModel.fit(synthetic_residuals)
    assert residual_pairs(holiday).empty


def test_counting_noise_flips_the_sign_of_the_residual_correlation():
    error_model = ErrorModel(sigma_eta=SIGMA_ETA, phi=PHI, tau2=TAU2)

    few_cases, many_cases = error_model.change_autocorrelation(np.array([5.0, 1e9]))

    assert few_cases < 0  # counting noise: a week reported too high is followed by an apparent drop
    assert many_cases == pytest.approx(PHI)
    assert error_model.change_std(1e9) == pytest.approx(SIGMA_ETA)


def test_process_error_is_stationary_with_persistence_phi():
    error_model = ErrorModel(sigma_eta=SIGMA_ETA, phi=PHI, tau2=TAU2)
    rng = np.random.default_rng(1)
    eta = [error_model.initial_process_error(20_000, rng)]
    for _ in range(3):
        eta.append(error_model.next_process_error(eta[-1], rng))

    assert np.std(eta[-1]) == pytest.approx(SIGMA_ETA, rel=0.03)
    assert np.corrcoef(eta[-2], eta[-1])[0, 1] == pytest.approx(PHI, abs=0.03)


def test_counting_noise_shrinks_with_the_number_of_cases():
    error_model = ErrorModel(sigma_eta=SIGMA_ETA, phi=PHI, tau2=TAU2)
    rng = np.random.default_rng(2)

    assert np.std(error_model.counting_noise(np.full(20_000, 100.0), rng)) == pytest.approx(0.1, rel=0.03)
    assert np.std(error_model.counting_noise(np.full(20_000, 0.01), rng)) == pytest.approx(1.0, rel=0.03)


def test_error_model_survives_the_round_trip_through_a_dict():
    error_model = ErrorModel(sigma_eta=SIGMA_ETA, phi=PHI, tau2=TAU2)

    assert ErrorModel.from_dict(error_model.to_dict()) == error_model
