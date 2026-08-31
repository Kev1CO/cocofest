from __future__ import annotations

import numpy as np
from casadi import MX
from scipy import optimize

from cocofest import DingModelPulseWidthFrequency

# Fixed
R0 = 2.0  # Value from Kesar et al. (2008)
TAUC = 0.020  # Value in (s) from Kesar et al. (2008)
SUM_STIM_TRUNCATION = 10
A_DEFAULT = 900.0


class DingModelArticle(DingModelPulseWidthFrequency):
    """
    Ding, J., Chou, L. W., Kesar, T. M., Lee, S. C., Johnston, T. E., Wexler, A. S., & Binder-Macleod, S. A.
    (2007). Mathematical model that predicts the force-intensity and force-frequency relationships after
    spinal cord injuries. Muscle & Nerve, 36(2), 214-222.
    """

    def __init__(
        self,
        model_name: str = "ding_2007_article",
        muscle_name: str = None,
        stim_time: list[float] = None,
        previous_stim: dict = None,
        sum_stim_truncation: int = SUM_STIM_TRUNCATION,
        **identified,
    ):
        super().__init__(
            model_name=model_name,
            muscle_name=muscle_name,
            stim_time=stim_time,
            previous_stim=previous_stim,
            sum_stim_truncation=sum_stim_truncation,
        )
        self.tauc = TAUC
        self.r0 = R0
        self.hold(**identified)

    def hold(self, **identified):
        """
        Hold every named parameter at the value an earlier step settled it to.
        """
        for name, value in identified.items():
            if not hasattr(self, name):
                raise TypeError(f"{type(self).__name__} carries no parameter named {name!r}")
            if value is not None:
                setattr(self, name, value)

    def get_r0(self, km: MX | float) -> MX | float:
        """The R0 term, which the article fixes"""
        return self.r0


class DingModelSingleIntensity(DingModelArticle):
    """
    The same model with A as a free scalar rather than a function of the pulse width.

    Steps 1 to 3 run at a single intensity and identify A itself; eq. 3 only comes in at step 4. A is
    carried on its own attribute rather than on the parent's a_scale, which is the eq. 3 asymptote a' and
    equals A only at an infinite pulse width.
    """

    def __init__(self, A: float = A_DEFAULT, **kwargs):
        super().__init__(**kwargs)
        self.A = A

    def set_A(self, model, A: MX | float):
        """Set A, the force-scaling factor of this stimulation intensity (N/s)."""
        self.A = A

    def a_calculation(self, a_scale: float | MX, pulse_width: MX) -> MX:
        """The force-scaling factor A (N/s), which is the identified scalar itself here."""
        return self.A


# ---- Equation 3, the force-intensity relationship ---- #
def a_calculation(pulse_width, a_scale, pd0, pdt):
    """
    Eq. 3 on numpy arrays: A (N/s) at a pulse width (s), given the asymptote a', pd0 and pdt.
    """
    return a_scale * (1 - np.exp(-(np.asarray(pulse_width, dtype=float) - pd0) / pdt))


A_SCALE_HEADROOM = 20.0
PD0_FLOOR = 50e-6


def profile_pd0(pulse_widths, a_values, points=15) -> dict:
    """
    How much the eq. 3 fit depends on pd0, by holding it and refitting a' and pdt at each value.

    A minimum inside the range means the intensities carry a threshold; a profile that only rises from zero
    means the reported pd0 is the floor rather than a measurement. The grid starts below PD0_FLOOR so a fit
    pinned by the floor shows as such.

    Returns
    -------
    dict
        The pd0 grid, the sum of squares and the R2 at each, the best pd0, and whether it sits inside
    """
    pulse_widths = np.asarray(pulse_widths, dtype=float)
    a_values = np.asarray(a_values, dtype=float)
    grid = np.linspace(0.0, 0.95 * pulse_widths.min(), points)
    spread = float(np.sum((a_values - a_values.mean()) ** 2))

    residuals = []
    for pd0 in grid:
        fit = optimize.least_squares(
            lambda p, pd0=pd0: a_calculation(pulse_widths, p[0], pd0, p[1]) - a_values,
            [1.2 * a_values.max(), 1.94138e-4],
            bounds=([a_values.max(), 1e-6], [A_SCALE_HEADROOM * a_values.max(), 5e-3]),
        )
        residuals.append(float(np.sum(fit.fun**2)))

    sse = np.array(residuals)
    best = int(np.argmin(sse))
    return {
        "pd0": grid,
        "sse": sse,
        "r2": 1 - sse / spread if spread > 0 else np.full(len(sse), np.nan),
        "best": float(grid[best]),
        "floor": PD0_FLOOR,
        "interior": bool(0 < best < len(grid) - 1),
    }


def fit_a_pulse_width(pulse_widths, a_values, limits=None) -> dict:
    """
    Step 4 of the article: fit eq. 3 to the A identified at each intensity.

    Returns
    -------
    dict
        a_scale, pd0 and pdt, the bounds the fit was given, its R2, and whether A rose with the pulse width
    """
    pulse_widths = np.asarray(pulse_widths, dtype=float)
    a_values = np.asarray(a_values, dtype=float)
    if len(pulse_widths) < 3:
        raise ValueError(f"Eq. 3 has three parameters, {len(pulse_widths)} intensities cannot pin them down.")

    lower = [a_values.max(), min(PD0_FLOOR, 0.5 * 0.95 * pulse_widths.min()), 1e-6]
    upper = [A_SCALE_HEADROOM * a_values.max(), 0.95 * pulse_widths.min(), 5e-3]
    if limits:
        lower = [max(a_values.max(), limits["a_scale"][0]), limits["pd0"][0], limits["pdt"][0]]
        upper = [max(limits["a_scale"][1], 1.001 * lower[0]),
                 min(limits["pd0"][1], 0.95 * pulse_widths.min()), limits["pdt"][1]]
    guess = list(np.clip([1.2 * a_values.max(), 1.31405e-4, 1.94138e-4], lower, upper))

    fit = optimize.least_squares(
        lambda p: a_calculation(pulse_widths, *p) - a_values, guess, bounds=(lower, upper)
    )
    residual = a_calculation(pulse_widths, *fit.x) - a_values
    spread = float(np.sum((a_values - a_values.mean()) ** 2))
    return {
        "a_scale": float(fit.x[0]),
        "pd0": float(fit.x[1]),
        "pdt": float(fit.x[2]),
        "bounds": dict(zip(("a_scale", "pd0", "pdt"), zip(lower, upper))),
        "r2": float(1 - np.sum(residual**2) / spread) if spread > 0 else np.nan,
        "monotonic": bool(np.all(np.diff(a_values[np.argsort(pulse_widths)]) >= 0)),
    }
