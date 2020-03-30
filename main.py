#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 17 15:17:00 2019
Python V3.7.1

@author: Job Formsma
"""

import numpy as np
import matplotlib.pyplot as plt

from mlmodel.system import MLsystem


def main():

    # Silicon properties
    dielectric_si = 11.68
    loss_tan_si = 1e-4

    # Full dielectric constant of Silicon including loss
    e_si = dielectric_si * (1 - 1j * loss_tan_si)

    # Create a system object surrounded by air (e=1)
    system = MLsystem(1)

    # Add individual layers of Fabry-Perot
    system.add("layer", 0.275e-3, e_si)          # Silicon waver
    system.add("layer", 0.680e-3, 1)             # Air gap
    system.add("layer", 0.275e-3, e_si)          # Silicon waver
    system.add("layer", 0.680e-3, 1)             # Air gap
    system.add("layer", 8e-3, e_si)              # Silicon slab
    system.add("layer", 0.680e-3, 1)             # Air gap
    system.add("layer", 0.275e-3, e_si)          # Silicon waver
    system.add("layer", 0.680e-3, 1)             # Air gap
    system.add("layer", 0.275e-3, e_si)          # Silicon waver

    # Radiation properties
    angle = [0]                                  # Incident angle(s)
    frequency = np.linspace(100e9, 110e9, 1000)  # Hz
    polarization = 's'

    # Calculate reflectance and transmittance
    R, T = system.RT(angle, frequency, polarization)
    A = 1 - R - T                                # Absorption

    # Plot Results
    plotter(frequency, R, T, A)


def plotter(F, R, T, A):

    # Show Results
    fig, frames = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    frames[0].plot(F * 1e-9, R, label='Reflectance')
    frames[1].plot(F * 1e-9, T, label='Transmittance', c='g')
    frames[2].plot(F * 1e-9, A, label='Absorption', c='orange')

    frames[2].set_xlabel("Frequency (GHz)")

    for frame in frames:
        frame.set_ylabel("Response")
        frame.set_ylim([0, 1])
        frame.grid(True)
        frame.legend(loc=1)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
