"""
The Ding 2007 model of cocofest, initialised the way the article fixes it, and the child the staged
identification needs.

    "Pilot testing of the model showed that parameter R0 could be fixed at 5 and parameter tauc could be
     fixed at 11 ms for the quadriceps femoris muscles of individuals with SCI."

Those two numbers are a quadriceps under spinal cord injury, and neither transfers to a healthy biceps by
assumption. tauc does transfer as a *rule* rather than as a value: the article's own earlier work gets it
from the muscle's relaxation, "calculated by multiplying the subject's half-relaxation time by 0.22", which
is what gave 11 ms there. This protocol records no single-pulse twitch, so that rule cannot be applied as
the article applies it, and both are held at the Kesar et al. (2008) values below.

cocofest inherits Ding 2003's R0 = Km + 1.04 instead, so `DingModelArticle` overrides `get_r0`. Nothing
else of the dynamics is touched: eqs. 1 and 2 stay the ones in cocofest/models/ding2007/ding2007.py. What
a step settles is handed to the next one through the constructor, see `hold`.
"""

from __future__ import annotations

import numpy as np
from casadi import MX
from scipy import optimize

from cocofest import DingModelPulseWidthFrequency

# Fixed
R0 = 2.0  # Value from Kesar et al. (2008)
TAUC = 0.020  # Value in (s) from Kesar et al. (2008)
SUM_STIM_TRUNCATION = 10

# Where A starts before a step identifies it, in the range this cohort's biceps produces (N/s)
A_DEFAULT = 900.0

# tau2 is held here rather than identified. Free, it fails in both routes and in opposite directions: the
# flexion pins it to its lower bound on 19 of 20 participants, and the force puts it at a median 53.9 ms
# with a CV above 100 %, fifty times the literature value, buying plateau accuracy with a relaxation that is
# far too slow. Neither is a measurement, so it is fixed the way the article fixes R0 and tauc.
TAU2_LITERATURE = 1e-3  # Value from Ding's 2007 article (s)


class DingModelArticle(DingModelPulseWidthFrequency):
    """
    The Ding 2007 model with R0 and tauc held where the article holds them.

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

        This is the door the staged identification needs: what a step does not identify still has to reach
        the model, and a step should be able to hand over the whole parameter dict of the step before it
        rather than name each parameter again. What is not handed over keeps cocofest's value.

        Parameters
        ----------
        identified: dict
            Parameter name to value, ex: the `parameters` a previous step returned. A name the model does
            not carry is an error rather than a new attribute, so a typo cannot pass for a held parameter.
        """
        for name, value in identified.items():
            if not hasattr(self, name):
                raise TypeError(f"{type(self).__name__} carries no parameter named {name!r}")
            if value is not None:
                setattr(self, name, value)

    def get_r0(self, km: MX | float) -> MX | float:
        """
        Compute the R0 term, which the article fixes rather than ties to Km.

        Parameters
        ----------
        km: MX | float
            The current cross-bridges sensitivity to Cn, unused here

        Returns
        -------
        MX | float
            The R0 term (magnitude of enhancement in Cn from the following stimuli)
        """
        return self.r0


class DingModelSingleIntensity(DingModelArticle):
    """
    The same model with A as a free scalar rather than a function of the pulse width.

    Steps 1 and 2 of the article each run at a single stimulation intensity and identify A itself. Eq. 3
    only comes in at step 4, once one A per intensity is available, so it is stepped over here.

    A is carried on its own attribute rather than on a_scale, which the parent uses for the eq. 3 asymptote
    a'. The two are the same number only at an infinite pulse width, so a step that identifies "A" and a step
    that identifies "a_scale" now say so in their parameter name, and neither can be read as the other.
    """

    def __init__(self, A: float = A_DEFAULT, **kwargs):
        super().__init__(**kwargs)
        self.A = A

    def set_A(self, model, A: MX | float):
        """Set A, the force-scaling factor of this stimulation intensity (N/s)."""
        self.A = A

    def a_calculation(self, a_scale: float | MX, pulse_width: MX) -> MX:
        """
        Compute the force-scaling factor A, which is the identified scalar itself.

        Parameters
        ----------
        a_scale: float | MX
            The eq. 3 asymptote the parent reads off the instance, unused here
        pulse_width: MX
            The pulsation duration of the current stimulation (s), unused here

        Returns
        -------
        The value of scaling factor (N/s)
        """
        return self.A


# ---- Equation 3, the force-intensity relationship ---- #
def a_calculation(pulse_width, a_scale, pd0, pdt):
    """
    Compute the pulse-width-scaled force-scaling factor A, on numpy arrays (eq. 3).

    The model's own `a_calculation` is symbolic and reads pd0/pdt off the instance. This one takes them as
    arguments, which is what fitting eq. 3 and drawing it need.

    Parameters
    ----------
    pulse_width: float | np.ndarray
        The pulsation duration of the stimulation (s)
    a_scale: float
        The asymptote A tends to at long pulse widths (N/s)
    pd0: float
        The pulse width below which the muscle produces no force (s)
    pdt: float
        The pulse width scale over which A rises (s)

    Returns
    -------
    The value of scaling factor (N/s)
    """
    return a_scale * (1 - np.exp(-(np.asarray(pulse_width, dtype=float) - pd0) / pdt))


# How far above the largest identified A the eq. 3 asymptote may sit. A few intensities spanning 250-500 us
# cannot see an asymptote further out than that, and a wider range only lets the fit run away.
A_SCALE_HEADROOM = 20.0

# The smallest pd0 the fit may return (s). pd0 is the pulse width below which the muscle produces no force,
# and this protocol records nothing below 250 us, so the threshold is extrapolated rather than measured. A
# route whose A values are too flat drives it to zero, which claims the muscle answers any pulse width at
# all; the floor keeps the fit physically admissible and makes the pinning visible through `at_bound`.
PD0_FLOOR = 50e-6


def profile_pd0(pulse_widths, a_values, points=15) -> dict:
    """
    How much the eq. 3 fit actually depends on pd0, by holding it and refitting a' and pdt at each value.

    The bound check tells you pd0 landed on its floor; this tells you whether that is the data speaking or
    the floor speaking. A profile with a minimum inside the range means the intensities carry a threshold; a
    profile that only rises from zero means they do not, and the pd0 reported is the floor, not a
    measurement. The grid deliberately starts at zero, below PD0_FLOOR, so a fit pinned by the floor is
    visible as such.

    Parameters
    ----------
    pulse_widths: list | np.ndarray
        The identification pulse widths (s)
    a_values: list | np.ndarray
        The A identified at each of them
    points: int
        How many pd0 values the profile is taken at

    Returns
    -------
    dict
        The pd0 grid, the sum of squares and the R2 at each, the best pd0, and whether that best sits
        strictly inside the grid
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
        # A minimum on either edge is the grid choosing, not the data
        "interior": bool(0 < best < len(grid) - 1),
    }


def fit_a_pulse_width(pulse_widths, a_values) -> dict:
    """
    Step 4 of the article: fit eq. 3 to the A identified at each intensity.

    Parameters
    ----------
    pulse_widths: list | np.ndarray
        The identification pulse widths (s), in the order a_values is given
    a_values: list | np.ndarray
        The A identified at each of them (N/s)

    Returns
    -------
    dict
        a_scale, pd0 and pdt, the bounds the fit was given, its R2, and whether A rose with the pulse width
    """
    pulse_widths = np.asarray(pulse_widths, dtype=float)
    a_values = np.asarray(a_values, dtype=float)
    if len(pulse_widths) < 3:
        raise ValueError(f"Eq. 3 has three parameters, {len(pulse_widths)} intensities cannot pin them down.")

    # Bounds read off the A values, not off the literature: this cohort's biceps produces an order of
    # magnitude less force than Ding's quadriceps. a' is bounded below by the largest A observed, since
    # eq. 3 approaches it from below without ever reaching it.
    lower = [a_values.max(), min(PD0_FLOOR, 0.5 * 0.95 * pulse_widths.min()), 1e-6]
    upper = [A_SCALE_HEADROOM * a_values.max(), 0.95 * pulse_widths.min(), 5e-3]
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
        # Eq. 3 rises with the pulse width and can do nothing else. A values that do not are a subject whose
        # intensity effect sits under the noise, and the three parameters fitted to them mean nothing.
        "monotonic": bool(np.all(np.diff(a_values[np.argsort(pulse_widths)]) >= 0)),
    }

