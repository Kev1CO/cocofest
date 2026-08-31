"""
The experimental stimulation conditions and the split method.

    freq \\ pw    250   300   350   400   450   500
    20 Hz          o     x     o     R     o     x
    33 Hz          o     o     x     R     o     o
    50 Hz          x     o     o     R     x     o

    R  reference intensity, the one column kept whole: tau1 at step 1, then A, Km and tau2 at step 2
    o  identification: A re-identified per pulse width (step 3), then eq. 3 through those A (step 4)
    x  predictive testing, never seen by the identification (step 5)

13 identification trains and 5 testing. Three rules hold the table together and `check_design` enforces
them: two identification trains per pulse width, one testing train per frequency, and testing trains
spread over five different pulse widths.
"""

from __future__ import annotations

REFERENCE, IDENTIFICATION, TESTING = "R", "o", "x"
DROPPED = "-"

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
    """A pulse width in microseconds, whichever unit it is handed in: below 1 it is read as seconds."""
    value = float(pulse_width)
    return value * 1e6 if value < 1.0 else value


def nearest(value, grid) -> float:
    """The grid value a recorded condition belongs to."""
    return min(grid, key=lambda candidate: abs(candidate - value))


def role(frequency, pulse_width, design=None) -> str:
    """Whether one train is the reference intensity, an identification train, or a testing train."""
    row = (design or DESIGN)[nearest(float(frequency), FREQUENCIES)]
    return row[nearest(as_microseconds(pulse_width), PULSE_WIDTHS_US)]


def condition_name(frequency, pulse_width) -> str:
    """How one stimulation condition is named in the tables and figures, ex: "50Hz 400us"."""
    return f"{int(round(float(frequency)))}Hz {as_microseconds(pulse_width):.0f}us"


def split(frequencies, pulse_widths, design=None) -> tuple:
    """
    Sort the trains of one subject into the identification set and the testing set.

    Returns
    -------
    tuple
        The indices of the identification trains and the indices of the testing trains
    """
    identification, testing = [], []
    for i, (frequency, pulse_width) in enumerate(zip(frequencies, pulse_widths)):
        cell = role(frequency, pulse_width, design)
        if cell == DROPPED:
            continue
        (testing if cell == TESTING else identification).append(i)
    return identification, testing


def steps(frequencies, pulse_widths, design=None) -> tuple:
    """
    The trains each stage of the article's algorithm is run on.

    Returns
    -------
    tuple
        The indices of the reference intensity trains (steps 1 and 2), and an ordered list of
        (pulse width in us, indices) for the other identification intensities (step 3)
    """
    reference, others = [], {}
    for i, (frequency, pulse_width) in enumerate(zip(frequencies, pulse_widths)):
        cell = role(frequency, pulse_width, design)
        if cell == REFERENCE:
            reference.append(i)
        elif cell == IDENTIFICATION:
            others.setdefault(nearest(as_microseconds(pulse_width), PULSE_WIDTHS_US), []).append(i)
    return reference, sorted(others.items())


def grid_table(frequencies=None, pulse_widths=None) -> str:
    """The design as the article prints its Table 1, a condition one subject lacks printed "."."""
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
    """The rules of the module docstring that DESIGN breaks, one line each, empty when it holds."""
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


def report_split(frequencies, pulse_widths, label="", design=None) -> None:
    """Print how one subject's trains were split."""
    identification, testing = split(frequencies, pulse_widths, design)
    print(f"{label}{len(identification)} identification / {len(testing)} testing trains")
    for name, indices in (("identification", identification), ("testing", testing)):
        conditions = [condition_name(frequencies[i], pulse_widths[i]) for i in indices]
        print(f"  {name:>15}: {', '.join(conditions) if conditions else 'none'}")


if __name__ == "__main__":
    print(grid_table())
