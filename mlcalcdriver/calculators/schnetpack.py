r"""
Calculator metaclass to accomodate machine learning models
trained using the SchnetPack package.
"""

import os
import numpy
import torch
from mlcalcdriver.calculators import Calculator


class SchnetPackCalculator(Calculator):
    r"""
    Calculator based on a SchnetPack model
    """

    def __init__(self, model_dir, available_properties=None, device="cpu"):
        r"""
        Parameters
        ----------
        model_dir : str
            Path to the stored model on which the calculator
            will be based. If $MODELDIR is defined, the path can
            be relative to it. If not, the path must be absolute
            or relative to the working directory.
        available_properties : str or list of str
            Properties that the model can predict. If `None`, they
            automatically determined from the model. Default is `None`.
        device : str
            Can be either `"cpu"` to use cpu or `"cuda"` to use "gpu"
        """
        try:
            self.model = load_model(
                model_dir=os.environ["MODELDIR"] + model_dir, device=device
            )
        except:
            self.model = load_model(model_dir=model_dir, device=device)
        super(SchnetPackCalculator, self).__init__()

    def run(self, property, posinp=None):
        r"""
        Main method to use when making a calculation with
        the calculator.
        """
        if property not in self.available_properties:
            raise ValueError(
                    "The property {} is not in the available properties of the model : {}.".format(
                    property, self.available_properties
                )
            )
            
    def _get_available_properties(self):
        r"""
        Returns
        -------
        avail_prop
            Properties that the SchnetPack model will return
        """
        avail_prop = set([om.property for om in self.model.output_modules])
        return avail_prop


def load_model(model_dir, device):
    try:
        model_dir = str(model_dir)
    except Exception as e:
        raise e
    try:
        model = torch.load(model_dir, map_location=device)
    except Exception as e:
        raise e
    return model
