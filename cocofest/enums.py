from enum import Enum


class StimulationMode(Enum):
    """
    Selection of valid stimulation types.
    """

    SINGLE = "single"
    DOUBLET = "doublet"
    TRIPLET = "triplet"
