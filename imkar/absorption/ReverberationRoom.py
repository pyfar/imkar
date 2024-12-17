"""A module to implement the ISO 354.
"""
import pyfar as pf
import numpy as np


class ReverberationRoom:
    """A class to check the conditions for a reverberation room according to
    ISO 354.

    Parameters
    ----------
    volume : float
        the volume of the room, in cubic metres.
    surface : float
        _description_
    I_max : float
        the length of the longest straight line which fits within the
        boundary of the room (e.g. in a rectangular room it is the
        major diagonal), in metres.
    """

    def __init__(self, volume, surface, I_max):
        """Initialize .

        Parameters
        ----------
        volume : float
            the volume of the room, in cubic metres.
        surface : float
            _description_
        I_max : float
            the length of the longest straight line which fits within the
            boundary of the room (e.g. in a rectangular room it is the
            major diagonal), in metres.
        """
        self.volume = volume
        self.surface = surface
        self.I_max = I_max

    def check_geometrically(self):
        """Check the geometric conditions for the reverberation room.

        - The volume of the room shall be greater than 150 m^3.
        - The volume of the room shall be less than 500 m^3.
        - The length of the longest straight line which fits within the
          boundary of the room shall be less than 1.9 times the cubic root of
          the volume of the room.

        Parameters
        ----------
        volume : float
            the volume of the room, in cubic metres.
        I_max : float
            the length of the longest straight line which fits within the
            boundary of the room (e.g. in a rectangular room it is the
            major diagonal), in metres.
        """
        # 6.1.1: V > 150, better V > 200, V < 500
        if self.volume < 150:
            raise ValueError(
                "The volume of the room shall be greater than 150 m^3.")
        if self.volume > 500:
            raise ValueError(
                "The volume of the room shall be less than 500 m^3.")
        # 6.1.2: I_max < 1.9 V ** (1/3)
        if self.I_max > 1.9 * self.volume ** (1/3):
            raise ValueError(
                "The length of the longest straight line which fits within "
                "the boundary of the room shall be less than 1.9 times the "
                "cubic root of the volume of the room.")
        return True # all conditions are fulfilled

    def check_iso(
            self, reverberation_time, speed_of_sound, attenuation_factor):
        """Check weather the conditions for the equivalent absorption area
        are fulfilled.

        - The graph of the equivalent sound absorption area of the empty room
          versus the frequency shall be a smooth curve and shall have no dips
          or peaks differing by more than 15 % from the mean of the values
          of both adjacent one-third-octave bands.
        - The equivalent sound absorption area of the empty room shall be
          smaller than the maximum equivalent sound absorption area.

        Parameters
        ----------
        reverberation_time : pf.FrequencyData
            the reverberation time, in seconds,
            of the empty reverberation room;
        speed_of_sound : float
            the propagation speed of sound in air, in metres per second.
        attenuation_factor : float
            is the power attenuation coefficient, in reciprocal metres,
            calculated according to ISO 9613-1 using the climatic conditions
            that have been present in the empty reverberation room during
            the measurement. The value of m can be calculated from the
            attenuation coefficient, alpha, which is used in ISO 9613-1
            according to the formula.

        Returns
        -------
        valid_results : array of bools
            whether the conditions are fulfilled for each frequency band.
        A_1 : pf.FrequencyData
            the equivalent absorption area of the empty room.
        max_A : pf.FrequencyData
            the maximum equivalent absorption area.
        """
        A_1 = 55.3 * self.volume / (speed_of_sound * reverberation_time) - (
            4 * self.volume * attenuation_factor)
        max_A = self.maximum_equivalent_absorption_area()

        # Check if A_1 is smaller than max_A
        valid_results = A_1 < max_A

        # Check for dips or peaks differing by more than 15%
        for i in range(1, len(A_1) - 1):
            mean_adjacent = (A_1[i - 1] + A_1[i + 1]) / 2
            if abs(A_1[i] - mean_adjacent) > 0.15 * mean_adjacent:
                valid_results[i] = False

        return valid_results, A_1, max_A

    def maximum_equivalent_absorption_area(self):
        """Calculate the maximum equivalent absorption area according to ISO.

        A correction for the volume of the room is applied if the volume is not
        200 m^3.

        Returns
        -------
        equivalent_absorption_area : pf.FrequencyData
            The equivalent absorption area
        """
        equivalent_absorption_area = pf.FrequencyData(
            [
                100, 125, 160, 200, 250, 315, 400, 500, 630,
                800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000,
            ],
            [
                6.5, 6.5, 6.5, 6.5, 6.5, 6.5, 6.5, 6.5, 6.5,
                6.5, 7.0, 7.5, 8.0, 9.5, 10.5, 12.0, 13.0, 14.0,
            ])
        if np.abs(self.volume - 200) > 5:
            equivalent_absorption_area = equivalent_absorption_area * (
                self.volume/200)**(2/3)

        return equivalent_absorption_area

