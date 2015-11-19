import pandas as pd
import abc
import pkg_resources as pkg
from numpy import *
from scipy.interpolate import griddata

class BuildupParameterTable(object):
    def __init__(self, data):
        self._data = data

        self.points = map(asarray, asarray(data.index))
        self.vals = asarray(data)

    @classmethod
    def load_coefficient_data(cls, path):
        with pd.HDFStore(path) as store:
            data = {}
            for key in store.keys():
                newkey = key.replace("/", "", 1)
                data[newkey] = cls(store[key])

        return data

    def __getitem__(self, key):
        return griddata(self.points, self.vals, atleast_2d(key), method='cubic').flatten()

    def __repr__(self): # Return the underlying dataframe's repr
        return self._data.__repr__()

class BuildupModelBase(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, materials):
        self.materials = materials

    def __repr__(self):
        return "\n\n".join(
            [
                "=========== {mat} ===========\n\n{repr}".format(mat=m, repr=s.__str__())
                for m, s in self.materials.iteritems()
            ]
        )

    @abc.abstractmethod
    def __call__(self):
        return NotImplemented

class HarimaModel(BuildupModelBase):
    def __call__(self, m, x, e0):
        b, c, a, xk, d = self.materials[m][e0]

        if x <= xk:
            k = c * (x ** a)
        else:
            k = c * (x ** a) + d * (x - xk)

        if isclose(k, 1.0):
            return 1. + (b - 1.) * x
        else:
            return 1. + ((b - 1.) / (k - 1)) * ((k ** x) - 1)

# Load up the Harima coefficients and initialize
harima_fname = pkg.resource_filename("gefry2", "data/HarimaCoefficients.h5")
harima_data = BuildupParameterTable.load_coefficient_data(harima_fname)
harima_buildup_model = HarimaModel(harima_data)
