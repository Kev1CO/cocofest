"""
The stimulation conditions of this cohort, split the way Ding et al. (2007) splits its Table 1.

    "The 50-Hz CFTs and 20-Hz DFTs at 150, 200, 300, and 500 us were used to determine the model's
     parameter values for each subject. The remaining trains were used to test the model's predictive
     ability."

The article identifies at one reference intensity and re-identifies A alone at the others, then predicts
whatever is left. The design below keeps that structure over the 3 x 6 grid recorded here.

    freq \\ pw    250   300   350   400   450   500
    20 Hz          o     x     o     R     o     x
    33 Hz          o     o     x     R     o     o
    50 Hz          x     o     o     R     x     o

    R  reference intensity: tau1 is read there (step 1), then A, Km and tau2 identified there (step 2)
    o  identification: A alone is re-identified there, one ocp per pulse width (step 3), and eq. 3 fitted
       through the A of every intensity (step 4)
    x  predictive testing: never seen by the identification (step 5)

13 identification trains, 5 testing, and no condition forgotten. Three rules fix it:

1.  Every pulse width keeps at least two identification trains, so eq. 3 is fitted on all six intensities
    and each of its A points comes from two trains rather than one.
2.  Every frequency carries at least one testing train, so the prediction is scored at 20, 33 and 50 Hz and
    not only where the trains happened to fall.
3.  The testing trains sit at five different pulse widths, so the prediction is scored across the intensity
    range instead of at a single one.

The reference intensity, 400 us, plays the role of the article's 300 us. It is the one column kept whole:
step 1 identifies the entire parameter set there, so it cannot afford to lose a frequency. It sits in the
upper half of the range on purpose, where the relaxation is large enough to read tau1 off.
"""

from __future__ import annotations

REFERENCE, IDENTIFICATION, TESTING = "R", "o", "x"

DESIGN = {
    20: {250: "o", 300: "x", 350: "o", 400: "R", 450: "o", 500: "x"},
    33: {250: "o", 300: "o", 350: "x", 400: "R", 450: "o", 500: "o"},
    50: {250: "x", 300: "o", 350: "o", 400: "R", 450: "x", 500: "o"},
}

FREQUENCIES = tuple(DESIGN)
PULSE_WIDTHS_US = tuple(next(iter(DESIGN.values())))
REFERENCE_PULSE_WIDTH_US = next(
    pulse_width for pulse_width, cell in DESIGN[FREQUENCIES[0]].items() if cell == REFERENCE
)


def as_microseconds(pulse_width) -> float:
    """
    A pulse width in microseconds, whichever unit it is handed in.

    The processed force files carry microseconds, the ocp data and the movement phases carry seconds, and a
    value below one is never a pulse width in microseconds.
    """
    value = float(pulse_width)
    return value * 1e6 if value < 1.0 else value


def nearest(value, grid) -> float:
    """The grid value a recorded condition belongs to."""
    return min(grid, key=lambda candidate: abs(candidate - value))


def role(frequency, pulse_width) -> str:
    """Whether one train is the reference intensity, an identification train, or a testing train."""
    row = DESIGN[nearest(float(frequency), FREQUENCIES)]
    return row[nearest(as_microseconds(pulse_width), PULSE_WIDTHS_US)]


def condition_name(frequency, pulse_width) -> str:
    """How one stimulation condition is named in the tables and figures, ex: "50Hz 400us"."""
    return f"{int(round(float(frequency)))}Hz {as_microseconds(pulse_width):.0f}us"


def split(frequencies, pulse_widths) -> tuple:
    """
    Sort the trains of one subject into the identification set and the testing set.

    The two routes hold their trains differently, one as the lists of an ocp data dictionary and the other
    as a list of movement phases, so every function here takes the two conditions as parallel sequences.

    Parameters
    ----------
    frequencies: list
        The stimulation frequency of every train (Hz)
    pulse_widths: list
        The pulse width of every train, in seconds or in microseconds

    Returns
    -------
    tuple
        The indices of the identification trains and the indices of the testing trains
    """
    identification, testing = [], []
    for i, (frequency, pulse_width) in enumerate(zip(frequencies, pulse_widths)):
        cell = role(frequency, pulse_width)
        (testing if cell == TESTING else identification).append(i)
    return identification, testing


def steps(frequencies, pulse_widths) -> tuple:
    """
    The trains each stage of the article's algorithm is run on.

    Steps 1 and 2 run on the reference intensity alone, not on the whole identification set: A is a single
    scalar there, and eq. 3 makes it vary by 1.86x across the recorded pulse widths.

    Returns
    -------
    tuple
        The indices of the reference intensity trains (steps 1 and 2), and an ordered list of
        (pulse width in us, indices) for the other identification intensities (step 3)
    """
    reference, others = [], {}
    for i, (frequency, pulse_width) in enumerate(zip(frequencies, pulse_widths)):
        cell = role(frequency, pulse_width)
        if cell == REFERENCE:
            reference.append(i)
        elif cell == IDENTIFICATION:
            others.setdefault(nearest(as_microseconds(pulse_width), PULSE_WIDTHS_US), []).append(i)
    return reference, sorted(others.items())


def grid_table(frequencies=None, pulse_widths=None) -> str:
    """
    The design, printed as the article prints its Table 1.

    Parameters
    ----------
    frequencies, pulse_widths: list
        The conditions actually available for one subject. The ones missing are printed ".".
    """
    available = {
        (nearest(float(frequency), FREQUENCIES), nearest(as_microseconds(pulse_width), PULSE_WIDTHS_US))
        for frequency, pulse_width in zip(frequencies or [], pulse_widths or [])
    }

    header = f"{'freq':>6}  " + "  ".join(f"{pulse_width:>5}" for pulse_width in PULSE_WIDTHS_US)
    lines = [header + "   (us)", "-" * len(header)]
    for frequency, row in DESIGN.items():
        cells = [
            "." if frequencies is not None and (frequency, pulse_width) not in available else cell
            for pulse_width, cell in row.items()
        ]
        lines.append(f"{frequency:>4}Hz  " + "  ".join(f"{cell:>5}" for cell in cells))

    identified = sum(1 for row in DESIGN.values() for cell in row.values() if cell != TESTING)
    lines += [
        "",
        "R = reference intensity, o = identification, x = predictive testing, . = not recorded",
        f"{identified} identification, {sum(len(row) for row in DESIGN.values()) - identified} testing",
    ]
    lines += [f"BROKEN RULE: {line}" for line in check_design()]
    return "\n".join(lines)


def check_design(minimum_per_pulse_width=2) -> list:
    """
    Check DESIGN against the three rules the module docstring states, and return what it breaks.

    Called by `grid_table`, so editing the table cannot silently drop a condition from eq. 3 or leave a
    frequency unscored.

    Returns
    -------
    list[str]
        One line per broken rule, empty when the design holds
    """
    broken = []

    for pulse_width in PULSE_WIDTHS_US:
        identified = sum(1 for row in DESIGN.values() if row[pulse_width] != TESTING)
        if identified < minimum_per_pulse_width:
            broken.append(f"{pulse_width} us is identified on {identified} train(s), rule 1 asks for "
                          f"{minimum_per_pulse_width}")

    for frequency, row in DESIGN.items():
        if TESTING not in row.values():
            broken.append(f"{frequency} Hz carries no testing train, rule 2 asks for one")

    tested = {pulse_width for row in DESIGN.values() for pulse_width, cell in row.items() if cell == TESTING}
    if len(tested) < len(PULSE_WIDTHS_US) - 1:
        broken.append(f"the testing trains sit at {len(tested)} pulse width(s), rule 3 asks for "
                      f"{len(PULSE_WIDTHS_US) - 1}")

    reference = [pulse_width for row in DESIGN.values() for pulse_width, cell in row.items() if cell == REFERENCE]
    if len(reference) != len(DESIGN) or len(set(reference)) != 1:
        broken.append("the reference intensity must be one whole column, step 1 needs every frequency of it")

    return broken


def report_split(frequencies, pulse_widths, label="") -> None:
    """Print how one subject's trains were split."""
    identification, testing = split(frequencies, pulse_widths)
    print(f"{label}{len(identification)} identification / {len(testing)} testing trains")
    for name, indices in (("identification", identification), ("testing", testing)):
        conditions = [condition_name(frequencies[i], pulse_widths[i]) for i in indices]
        print(f"  {name:>15}: {', '.join(conditions) if conditions else 'none'}")


if __name__ == "__main__":
    print(grid_table())
