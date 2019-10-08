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

    def __init__(self, available_properties):
        r"""
        Parameters
        ----------
        available_properties : str or list of str
            Properties that can be predicted by the model.
        """
        self.available_properties = available_properties 

    def run(self):
        r"""
        To be implemented for each type of model.
        """
        raise NotImplementedError

    @property
    def available_properties(self):
        r"""
        """
        return self._available_properties

    @available_properties.setter
    def available_properties(self, available_properties):
        self._available_properties = available_properties
