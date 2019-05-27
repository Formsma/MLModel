#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 17 13:36:06 2019
s
Python V3.7.1

@author: Job Formsma
"""

import sys
import numpy as np
import matplotlib.pyplot as plt

from mlmodel.system import MLsystem


class MLparser():
    """This class is built to easily implement the System
    class by parsing an input file with the system properties
    and incident radiation properties
    """

    def __init__(self, filename):
        """Initialize the class with the filename of the input
        file which needs to be in the documented format

        input:
            filename:    filename of input system
        """

        # Check if input exists
        try:
            self.file = open(filename, 'r')
        except Exception as e:
            raise e

        self.system = MLsystem()            # Initialize multilayer system
        self.mode = "INITIAL"               # Set initial parse mode
        self.N = 0                          # Line count
        self.param = []                     # Parametric plot counter

        # Setup default values for input radiation
        self.frequency = np.array([1e12])   # Default 1 THz
        self.angle = np.zeros(1)            # Default angle of zero
        self.polarization = 's'             # Default s polarization
        self.T = 293.2                      # Default room temperature

        # Start parsing of file
        try:
            self._read()
            # self._print()
        except Exception as e:
            print("Line {}: ".format(self.N) + str(e))
            sys.exit(1)

    def plot(self):
        """Run the simulation with the information gathered from
        the parsed file"""

        # Data storage for different polarizations
        storage = []

        # Run system according to the polarization input
        if (self.polarization == 's') or (self.polarization == 'sp'):

            # Calculate the reflectance and transmittance for s
            R, T = self.system.RTA(self.angle, self.frequency, 's')

            # Store data in case multiple polarizations are used
            storage.append([R, T, 1 - R - T])

        if (self.polarization == 'p') or (self.polarization == 'sp'):

            # Calculate the reflectance and transmittance for s
            R, T = self.system.RTA(self.angle, self.frequency, 'p')

            # Store data in case multiple polarizations are used
            storage.append([R, T, 1 - R - T])

        # Plot the data
        fig, frames = plt.subplots(3, 1, figsize=(15, 8), sharex=True)

        labels = ['Reflectance', 'Transmittance', 'Absorption']

        print("\n\nOutput...\n")

        # Iterate over polarization plots
        for i, data in enumerate(storage):

            # Clear color cycle
            for frame in frames:
                frame.set_prop_cycle(None)

            # If only a single data point, print instead of plot
            if len(self.param) == 0:

                print("Polarization {}: ".format(self.polarization[i]))
                print("R: {0:.4f}".format(*data[0][0]))
                print("T: {0:.4f}".format(*data[1][0]))
                print("A: {0:.4f}".format(*data[2][0]))
                return

            # If only 1 parametric input
            elif len(self.param) == 1:
                X_label = self.param[0][0]
                X = self.param[0][1]
                p = 1 if self.param[0][0] == 'deg' else 0

            # If two parametric inputs
            elif len(self.param) == 2:

                # Check which one is the X label
                p = 0 if len(self.param[0][1]) > len(self.param[1][1]) else 1

                # Assign labels
                X_label = self.param[p][0]
                X = self.param[p][1]
                P_label = self.param[-(p+1)][0]
                P = self.param[-(p+1)][1]

            # Plot data
            for dat, frame in zip(data, frames):

                # Transpose the data if the axis is not aligned
                dat = dat.T if not p else dat

                # Iterate over individual data points
                for k, d in enumerate(dat):

                    if len(self.param) == 2:
                        lbl = "{0:.2e} {1} {2}-polarized".format(
                                P[k], P_label, self.polarization[i])
                    else:
                        lbl = ''

                    # Plot data
                    frame.plot(X, d, ls='-'+'.'*i, label=lbl)

            # Set plot properties
            for j, frame in enumerate(frames):
                frame.set_title(labels[j])
                frame.grid(True)
                frame.set_xlim([X.min(), X.max()])
                if j == 2:
                    frame.set_xlabel(X_label)
                frame.set_ylim([0, 1])
                frame.set_ylabel(labels[j][0])
                frame.ticklabel_format(style='sci', axis='x',
                                       scilimits=(0, 0))
                if j == 0:
                        frame.legend(loc=1)

        plt.show()

    def _error(self, string):
        """Raise error which occurs in the class"""
        raise ValueError(string)

    def _read(self):
        """Read the input file by disregarding all unnecessary
        comments and read out the radiation and system properties
        """

        # Loop over input file
        for line in self.file:

            # Increase line count
            self.N += 1

            # Cut line into space seperated list
            line = line.split()

            # Skip all empty lines and comments
            if len(line) == 0:
                continue
            elif line[0][0] == '#':
                continue

            # If initial mode, do nothing until % is found
            if self.mode == "INITIAL":
                if line[0] == '%':
                    # The radiation mode identifier is found
                    self.mode = "RADIATION"

            # After a % is found, radiation mode is activated
            elif self.mode == "RADIATION":

                # Check for system token
                if line[0] == "%%":
                    # The system identifier is found
                    self.mode = "SYSTEM"
                    continue

                # Parse with radiation function on the line
                self._radiation_mode(line)

            # After a %% is found, system mode is activated
            elif self.mode == "SYSTEM":

                # Parse with system function on the line
                self._system_mode(line)

        # Check if final mode is system, otherwise input file incorrect
        if self.mode != "SYSTEM":
            self._error("Incorrect input file style, define radiation "
                        "and system with the '%' and '%%' identifiers")

    def _radiation_mode(self, line):
        """Extract information of the radiation that is inputted into
        the device. It has three modes: FREQUENCY, ANGLE and
        POLARIZATION to store the values
        """

        # If the line starts with the FREQUENCY token
        if line[0] == "FREQUENCY":
            self.frequency = self._parse(line[1:])
            self._param(self.frequency, "Hz")

        # If the line starts with the ANGLE token
        elif line[0] == "ANGLE":
            self.angle = self._parse(line[1:]) * np.pi / 180
            self._param(self.angle * 180 / np.pi, "deg")

        # If the line starts with the POLARIZATION token
        elif line[0] == "POLARIZATION":
            self.polarization = self._parse(line[1:])

        # If the line starts with the TEMPERATURE token
        elif line[0] == "TEMPERATURE":

            self._error("TEMPERATURE not yet supported")
            #self.T = self._parse(line[1:])
            #if not isinstance(self.T, float):
            #    self._error("Parametric input for temperature not supported")
            #self._param(self.T, "K")

        # If token fails
        else:
            # If token is missing look if can be converted to float
            try:
                float(line[0])
            except Exception:
                self._error("'{}' is not a correct token!".format(line[0]))

            # If a float is found instead of a token
            self._error("A token or '%%' identifier is missing!")

    def _parse(self, entry):
        """Parse the entry that is inputted for single value or
        look if input is in list shape"""

        # If single value input extract it
        if isinstance(entry, list):
            if len(entry) == 1:
                entry = entry[0]
            else:
                return np.linspace(*[float(i) for i in entry])

        # At this point we are sure the input is a string
        if entry.find('(') != -1:
            # Return a list of the exact input given
            return np.array([float(i) for i in entry[1:-1].split(',')])
        elif entry.find('[') != -1:
            # Return a linspace of the input range
            return np.linspace(*[float(i) for i in entry[1:-1].split(',')])
        else:
            # If polarization input
            if (entry == 's' or entry == 'p' or entry == 'sp'):
                return entry
            # Or just a single value, return float
            else:
                return float(entry)

    def _param(self, inp, name):
        """Check if input if parametric and store it"""

        if len(self.param) > 2:
            self._error("Too much parametric input")

        if not isinstance(inp, float):
            if len(inp) > 1:
                self.param.append((name, inp))

    def _system_mode(self, line):
        """Use the input to fill the system with layers, the add_layer
        fuction is called in every line provided in the input file.

        WIP: if a string is given with a filename the input is
        assumed to be a file instead of single float value
        """

        # Input data
        data = []

        # List for names
        names = ['m', "dielectric", "loss tangent"]
        # Loop over line contents to look for lists

        for i, entry in enumerate(line):
            entry = self._parse(entry)
            self._param(entry, names[i])
            data.append(entry)

        # Calculate complex dielectric constant
        e = data[1] * (1 - 1j * data[2])

        # Add layer to system
        if len(data) == 3:
            self.system.add_layer(data[0], e)
        #elif len(data) == 4:
        #    self.system.add_layer(data[0], e, data[3])
        else:
            self._error("Incorrect input for system layer")
