#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Fri May  3 15:33:36 2019.

Python V3.7.1

@author: Job Formsma
"""

import numpy as np
from scipy.interpolate import interp1d

from mlmodel.tlmatrix import TLmatrix

# Zero impedance of free space
Z_0 = 120 * np.pi


class MLsystem():
    """System class for multilayered systems.

    The MLsystem (multi-layered system)class is used for setting up
    a composition of slabs of dielectric material to simulate the
    reflection, transmission and absorption in such a system (i.e. a
    Fabry-Perot). Layers can be added to the system after which the
    complete system response is determined via the transmission line
    theory. The dielectric constant of the medium surrounding the
    system can be inputted at construction.
    """

    def __init__(self, e_surr=1):
        """Initialize empty lists for layer storage.

        args:
            e_surr:    dielectric constant of surrounding medium
        """

        # Incident and surrounding medium dielectric constant
        self.e_surr = e_surr

        # Layer storage
        self.type = []                # Layer type
        self.d = []                   # thickness of layer
        self.e = []                   # dielectric constant
        self.CTE = []                 # CTE component layer
        self.N = 0                    # amount of layers

    def add(self, layer_type=None, d=None, e=None, cte=None):
        """Store the inputted values in the class

        args:
            layer_type:    type of layer, i.e. dielectric or shunt
            d:             thickness layer
            e:             dielectric constant layer
            CTE:           CTE component layer
        """

        self.type.append(layer_type)  # store layer type
        self.d.append(d)              # add thickness to storage
        self.e.append(e)              # add dielectric constant
        self.CTE.append(cte)          # add CTE information
        self.N += 1                   # increase amount of layers

    def rt(self, angle, frequency, polarization, temperature=None):
        """Determine the reflectance and transmittance coefficient of
        the full system. The transmission line matrix of the full
        system is computed by multiplying the matrices of the
        individual layers.

        args:
            angle:          angle of incidence on the system
            frequency:      frequency of incident radiation
            polarization:   polarizatino of incident radiations
        returns:
            r:              reflection coefficient of system
            t:              transmission coefficient of the system
        """

        # Create identity matrix for multuiplication of the layers
        M = TLmatrix()

        # Loop over the layers to compute transmission line matrix of system
        for n in range(self.N):

            # Calculate thickness given temperature and CTE
            if temperature is not None:
                d = self._expansion(n, temperature)
            else:
                d = self.d[n]

            # Print layer info
            # self._info(n, d, self.e[n])

            # Construct and multiply a layer matrix to the system
            M @= TLmatrix(self.type[n], d, self.e[n], self.e_surr,
                          angle, frequency, polarization)

        # Compute source impedance
        Z_s = M.impedance(self.e_surr, angle, polarization)

        # Reflection coefficient
        r = ((M.A * Z_s + M.B - M.C * Z_s * Z_s - M.D * Z_s) /
             (M.A * Z_s + M.B + M.C * Z_s * Z_s + M.D * Z_s))

        # Transmission coefficient
        t = 2 * Z_s / (M.A * Z_s + M.B + M.C * Z_s * Z_s + M.D * Z_s)

        return r, t

    def RT(self, angle, frequency, polarization, temperature=None):
        """Determine the real reflectance and transmittance of
        the full system. The transmission line matrix of the full
        system is computed by multiplying the matrices of the
        individual layers.

        args:
            angle:          angle of incidence on the system
            frequency:      frequency of incident radiation
            polarization:   polarizatino of incident radiations
        returns:
            R:              reflectance of system
            T:              transmittance of the system
        """

        # Get field coefficients
        r, t = self.rt(angle, frequency, polarization, temperature)

        # Reflectance and Transmittance
        R = np.abs(r)**2
        T = np.abs(t)**2
        return R, T

    def _expansion(self, n, temperature):
        """Calculate the thickness of the layer which is altered
        due to a different temperature than the given reference
        thickness

        NOTE: THIS FUNCTION IS WIP AND RESULTS CAN BE WRONG

        args:
            n:              layer number
            temperature:    temperature of system
        returns:
            d:              thickness of layer
        """

        # Extract CTE components from database
        T, f = np.loadtxt("mlmodel/data/{}_t.dat".format(self.CTE[n]),
                          unpack=True)

        # Interpolate for the temperature of system
        return self.d[n] * (1 + interp1d(T, f)(temperature))

    def _info(self, n, d, e):
        """Print info of a system layer to the standard output"""

        print("Layer {}:".format(n))
        print("d:   ", d)
        print("e:   ", e)
        print("CTE: ", self.CTE[n])
        print()
