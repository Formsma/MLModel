#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  3 15:34:52 2019
Python V3.7.1

@author: Job Formsma
"""

import numpy as np

# Constants
c = 299792458           # Speed of light


class TLmatrix():
    """The TLmatrix class is used to build a transmission line matrix
    for a given layer of dielectric medium in a system
    """

    def __init__(self, layer_type=None, d=None, e=None, e_surr=None,
                 angle=None, frequency=None, polarization=None):
        """At construction the matrix elements are calculated according
        to the input layer and incident radiation. If no input is given
        the matrix will be an identity matrix.

        args:
            layer:          denotes layer type
            d:              thickness of medium
            e:              dielectric constant of the medium
            e_surr:         dielectric constant of surrounding medium
            angle:          angle of incidence of radiation
            frequency:      frequency of radiation through the medium
            polarization:   polarization of the incident radiation
        """
        # Check layer type
        if layer_type == "layer":

            # Calculate impedance
            Z = self.impedance(e, angle, polarization)

            # Compute transverse angle via Snell's law
            angle_t = np.arcsin(np.real(e_surr**0.5)
                                / np.real(e**0.5)
                                * np.sin(angle))

            # Go to wavelength space for next equations
            wavelength = c / frequency

            # Compute propagation constant
            gamma = 2j * np.outer(np.pi * e**0.5 / wavelength,
                                  np.cos(angle_t))

            # Compute matrix elements
            self.A = np.cosh(gamma * d)
            self.B = np.sinh(gamma * d) * Z
            self.C = np.sinh(gamma * d) / Z
            self.D = np.cosh(gamma * d)

        # If layer is a shunt element
        elif layer_type == "shunt":

            # Compute matrix elements
            self.A = 1
            self.B = 0
            self.C = 1 / d
            self.D = 1

        # If layer is a grid element
        elif layer_type == "grid" and polarization == 'p':

            # Go to wavelength space for next equations
            wavelength = c / frequency

            # Calculate grid impedance
            Z_0 = 120 * np.pi
            Z = 1j * Z_0 * np.outer(d / wavelength, np.log(1 / np.sin(np.pi * e / d)))

            # Correct for where Z is zero
            Z[np.where(Z == 0)[0]] += 1

            # Shunt like element
            self.A = 1
            self.B = 0
            self.C = 1 / Z
            self.D = 1


        # If no input is given the matrix will be a identity matrix
        else:
            self.A = 1
            self.B = 0
            self.C = 0
            self.D = 1

    def impedance(self, e, angle, polarization):
        """specific impedance < naam is neit correct"""

        # Free space zero impedance
        Z_0 = 120 * np.pi

        # Calculate impedance for both possible polarizations
        if polarization == 'p':
            return Z_0 * np.sqrt(e - np.sin(angle)**2) / e
        elif polarization == 's':
            return Z_0 / np.sqrt(e - np.sin(angle)**2)
        else:
            raise ValueError("Polarization input should be 's' or 'p'")

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
