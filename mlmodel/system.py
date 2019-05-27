#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Fri May  3 15:33:36 2019.

Python V3.7.1

@author: Job Formsma
"""

import numpy as np

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
    theory. The incident medium and last transverse medium is both air.
    """

    def __init__(self):
        """Initialize empty lists for layer storage"""

        # Layer storage
        self.d = []                 # thickness of layer
        self.e = []                 # dielectric constant
        self.N = 0                  # amount of layers

        # Temperature coefficients of the layers
        self.alpha = []

    def add_layer(self, d, e, a=0):
        """Add a layer of dielectric medium to the system. The
        thickness and dielectric constant are stored for later use.

        args:
            d:  thickness medium
            e:  dielectric constant medium
            a:  coefficient of thermal expansion at room temperature
        """

        self.d.append(d)            # add thickness to storage
        self.e.append(e)            # add dielectric constant
        self.alpha.append(a)        # CTE component layer
        self.N += 1                 # increase amount of layers

    def RTA(self, angle, frequency, polarization):
        """Determine the reflectance, transmittance and absorption of
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

        # Create identity matrix for multuiplication of the layers
        M = TLmatrix()

        # Loop over the layers to compute transmission line matrix of system
        for n in range(self.N):

            # Calculate thickness layer dependent on the system temperature
            d = self.d[n]                                                      # WORK IN PROGRESSS

            # Calculate refractive index dependent on system temperature
            e = self.e[n]                                                       # WORK IN PROGRESSS

            # Construct and multiply a layer matrix to the system
            M @= TLmatrix(d, e, angle, frequency, polarization)

        # Compute source impedance
        if polarization == 'p':
            Z_s = Z_0 * np.cos(angle)
        elif polarization == 's':
            Z_s = Z_0 / np.cos(angle)
        else:
            raise ValueError("Polarization input should be 's' or 'p'")

        # Reflection coefficient
        r = ((M.A * Z_s + M.B - M.C * Z_s * Z_s - M.D * Z_s) /
             (M.A * Z_s + M.B + M.C * Z_s * Z_s + M.D * Z_s))

        # Transmission coefficient
        t = 2 * Z_s / (M.A * Z_s + M.B + M.C * Z_s * Z_s + M.D * Z_s)

        # Reflectance and Transmittance
        R = np.abs(r)**2
        T = np.abs(t)**2

        return R, T

    def __str__(self):
        """Overload for print statements to nicely see the layers of
        the system

        returns:
            string: string consisting of matrix elements
        """
        # Start with basic information
        string = "System consists of {} layer{}\n\n".format(self.N, 's'*self.N)
        string += "|    | Thickness (m) | Dielectric constant |       CTE |\n"

        # Loop over layers to add information
        for n in range(self.N):

            # Add layer to string table
            string += "| {0:2} | {1} | {2:19} | {3:9} |\n".format(
                n, self.d[n], self.e[n], self.alpha[n]
            )

        return string
