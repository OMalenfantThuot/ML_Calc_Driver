r"""
"""

import numpy
from mlcalcdriver.calculators import Calculator


class SchnetPackCalculator(Calculator):
    r"""
    """
    def __init__(self, model_dir, available_properties=None):
        r"""
        """
        self.model_dir = model_dir
        self.available_properties = available_properties
