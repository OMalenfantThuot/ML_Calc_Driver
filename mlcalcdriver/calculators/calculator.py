from mlcalcdriver.globals import eVA

r"""
The :class:`Calculator` is the general class for a machine learning
calculator. A specific class derived from this one must be implemented
for each new type of model.
"""


class Calculator:
    r"""
    Class to be implemented individually for each type
    of models.
    """

    def __init__(self, available_properties=None, units=None):
        r"""
        Parameters
        ----------
        available_properties : str or list of str
            Properties that can be predicted by the Calculator. If `None`,
            the _get_available_properties method will be used.
        """
        if available_properties is None:
            self.available_properties = self._get_available_properties()
        else:
            self.available_properties = available_properties
        if units is None:
            self.units = self._get_units()
        else:
            self.units = units

    def run(self):
        r"""
        To be implemented for each type of model.
        """
        raise NotImplementedError

    def _get_available_properties(self):
        r"""
        To be implemented for each type of model
        """
        raise NotImplementedError

    def _get_units(self):
        r"""
        May be implemented for models for which it is possible.
        If not implemented, the units must be specified when creating
        the :class:`Calculator` instance.
        """
        raise NotImplementedError

    @property
    def available_properties(self):
        r"""
        Returns
        -------
        str or list of str
            Properties that can be predicted by the Calculator
        """
        return self._available_properties

    @available_properties.setter
    def available_properties(self, available_properties):
        self._available_properties = available_properties

    @property
    def units(self):
        r"""
        Returns
        -------
        dict:
            Dictionary containing the units used by the model
            keys() are `positions` and `energy`.
        """
        return self._units

    @units.setter
    def units(self, units):
        if isinstance(units, dict):
            if all(
                [k in ["positions", "energy", "dipole_moment"] for k in units.keys()]
            ):
                self._units = units
            else:
                raise KeyError("Units key not recognized.")
        else:
            raise TypeError("Units should be given in a dictionary.")


class DummyCalculator(Calculator):
    def __init__(self):
        properties = ["energy"]
        super().__init__(available_properties=properties, units=eVA)
