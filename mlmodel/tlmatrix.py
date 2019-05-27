#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  3 15:34:52 2019
Python V3.7.1

@author: Job Formsma
"""

import numpy as np

# Constants
Z_0 = 120 * np.pi       # Zero impedance of free space
c = 299792458           # Speed of light


class TLmatrix():
    """The TLmatrix class is used to build a transmission line matrix
    for a given layer of dielectric medium in a system
    """

    def __init__(self, d=None, e=None,
                 angle=None, frequency=None, polarization=None):
        """At construction the matrix elements are calculated according
        to the input layer and incident radiation. If no input is given
        the matrix will be an identity matrix.

        args:
            d:              thickness of medium
            e:              dielectric constant of the medium
            angle:          angle of incidence of radiation
            frequency:      frequency of radiation through the medium
            polarization:   polarization of the incident radiation
        """
        # If there is input calculate the matrix elements
        if d is not None:

            # Calculate impedance for both possible polarizations
            if polarization == 'p':
                Z = Z_0 * np.sqrt(e - np.sin(angle)**2) / e
            elif polarization == 's':
                Z = Z_0 / np.sqrt(e - np.sin(angle)**2)
            else:
                raise ValueError("Polarization input should be 's' or 'p'")

            # Compute transverse angle via Snell's law
            angle_t = np.arcsin(1 / np.real(e**0.5) * np.sin(angle))

            # Go to wavelength space for next equations
            wavelength = c / frequency

            # Compute propagation constant
            gamma = 2j * np.outer(np.pi * e**0.5 / wavelength,
                                  np.cos(angle_t))

            # Compute matrix elements
            self.A = np.cosh(gamma * d)                                   # uitzoeken welke transpose is en elke niet al gamma transpose dan meot d normaal zijn
            self.B = np.sinh(gamma * d) * Z
            self.C = np.sinh(gamma * d) / Z
            self.D = np.cosh(gamma * d)

        # If no input is given the matrix will be a identity matrix
        else:
            self.A = 1
            self.B = 0
            self.C = 0
            self.D = 1

    def __str__(self):
        """Overload for nice print statements"""
        return "A = {}\nB = {}\nC = {}\nD = {}\n".format(self.A, self.B,
                                                         self.C, self.D)

    def __imatmul__(self, other):
        """As the transmission line matrix can be high dimensional
        the imatmul operator for multiplying is overloaded to
        correctly calculate the new matrix, as the python
        implementation depends too much on dimensions.

        args:
            other:      another matrix object
        returns:
            self:       self object which is multiplied with other
        """

        # Calculate new matrix elements
        A = self.A * other.A + self.B * other.C
        B = self.A * other.B + self.B * other.D
        C = self.C * other.A + self.D * other.C
        D = self.C * other.B + self.D * other.D

        # Set the calculated values as self values before returning self
        self.A = A
        self.B = B
        self.C = C
        self.D = D

        return self
