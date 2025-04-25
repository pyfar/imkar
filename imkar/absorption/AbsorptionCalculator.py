"""A module to implement the ISO 354.
"""
import pyfar as pf
import numpy as np


class AbsorptionCalculator():
    """A class to check the conditions for a reverberation room according to
    ISO 354.

    Parameters
    ----------
    volume : float
        the volume of the room, in cubic metres.
    I_max : float
        the length of the longest straight line which fits within the
        boundary of the room (e.g. in a rectangular room it is the
        major diagonal), in metres.
    """

    def __init__(self, surface_area, volume, longest_dimension):
        """Initialize .

        Parameters
        ----------
        surface_area : float
            the surface area of the room, in square metres.
        volume : float
            the volume of the room, in cubic metres.
        longest_dimension : float
            the length of the longest straight line which fits within the
            boundary of the room (e.g. in a rectangular room it is the
            major diagonal), in metres.
        """
        self.volume = volume
        self.longest_dimension = longest_dimension
        self.surface_area = surface_area

    @classmethod
    def from_room_dimensions(cls, dimensions):
        """Create an AbsorptionCalculator from the dimensions of the room.

        Parameters
        ----------
        dimensions : tuple of floats
            the dimensions of the room in metres.
        """
        volume = dimensions[0] * dimensions[1] * dimensions[2]
        surface_area = 2 * (
            dimensions[0] * dimensions[1] + dimensions[0] * dimensions[2] +
            dimensions[1] * dimensions[2])
        longest_dimension = np.sqrt(
            dimensions[0]**2 + dimensions[1]**2 + dimensions[2]**2)
        return cls(surface_area, volume, longest_dimension)

    def set_measurement(
            self, measurement_empty, measurement_sample, sample_area):
        """Set the measurements of the empty room and the room with the sample.

        Parameters
        ----------
        measurement_empty : MeasurementResult
            Measurement of the empty room.
        measurement_sample : MeasurementResult
            Measurement of the room with the sample.
        sample_area : float
            is the area, in square metres, covered by the test specimen.
        """
        # test inputs

        # set attributes
        self.measurement_empty = measurement_empty
        self.measurement_sample = measurement_sample
        self.sample_area = sample_area
        pass

    def check_geometrically(self, verbose=True):
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
        verbose : bool, optional
            whether to print the conditions that are not fulfilled.
            The default is True.

        Returns
        -------
        conditions_fulfilled : bool
            whether all geometric conditions are fulfilled.
        """
        conditions_fulfilled = True
        # 6.1.1: V > 150, better V > 200, V < 500
        if self.volume < 150:
            if verbose:
                print(
                    "AbsorptionCalculator: The volume of the room "
                    "shall be greater than 150 m^3.")
            conditions_fulfilled = False
        if self.volume > 500:
            if verbose:
                print(
                    "AbsorptionCalculator: The volume of the room "
                    "shall be less than 500 m^3.")
            conditions_fulfilled = False
        # 6.1.2: I_max < 1.9 V ** (1/3)
        if self.I_max > 1.9 * self.volume ** (1/3):
            if verbose:
                print(
                    "AbsorptionCalculator: The length of the longest "
                    "straight line which fits within "
                    "the boundary of the room shall be less than 1.9 times "
                    "the cubic root of the volume of the room.")
            conditions_fulfilled = False
        return conditions_fulfilled

    def check_iso_empty(
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
        reverberation_time : :py:class:`~pyfar.classes.audio.FrequencyData`
            the reverberation time, in seconds,
            of the empty reverberation room;
        speed_of_sound : float
            the propagation speed of sound in air, in metres per second.
        attenuation_factor : :py:class:`~pyfar.classes.audio.FrequencyData`
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
        self.empty_reverberation_time = reverberation_time
        self.empty_speed_of_sound = speed_of_sound
        self.empty_attenuation_factor = attenuation_factor
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

    @staticmethod
    def maximum_equivalent_absorption_area(self):
        """Calculate the maximum equivalent absorption area according to ISO.

        A correction for the volume of the room is applied if the volume is not
        200 m^3.

        Returns
        -------
        equivalent_absorption_area : :py:class:`~pyfar.classes.audio.FrequencyData`
            The equivalent absorption area
        """  # noqa: E501
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

    def calculate_absorption(
                self, reverberation_time, speed_of_sound, attenuation_factor):
        """Calculate the absorption coefficient.

        Parameters
        ----------
        reverberation_time : :py:class:`~pyfar.classes.audio.FrequencyData`
            the reverberation time, in seconds,
            of the empty reverberation room;
        speed_of_sound : float
            the propagation speed of sound in air, in metres per second.
        attenuation_factor : :py:class:`~pyfar.classes.audio.FrequencyData`
            is the power attenuation coefficient, in reciprocal metres,
            calculated according to ISO 9613-1 using the climatic conditions
            that have been present in the empty reverberation room during
            the measurement. The value of m can be calculated from the
            attenuation coefficient, alpha, which is used in ISO 9613-1
            according to the formula.

        Returns
        -------
        absorption_coefficient : :py:class:`~pyfar.classes.audio.FrequencyData`
            the absorption coefficient.
        """
        self.filled_reverberation_time = reverberation_time
        self.filled_speed_of_sound = speed_of_sound
        self.filled_attenuation_factor = attenuation_factor
        A_1 = 55.3 * self.volume / (
            self.empty_speed_of_sound * \
                self.empty_reverberation_time) - (
            4 * self.volume * self.empty_attenuation_factor)
        A_2 = 55.3 * self.volume / (
            speed_of_sound * \
                reverberation_time) - (
            4 * self.volume * attenuation_factor)
        A_T = A_2 - A_1
        absorption_coefficient = A_T / self.sample_area
        return absorption_coefficient

    def analyse_environmental_conditions(
            self, temperature, humidity, method='ISO'):
        """Analyse the environmental conditions of the room.

        Parameters
        ----------
        temperature : float
            the temperature in degrees Celsius.
        humidity : float
            the relative humidity in percent.
        method : str, optional
            the method to calculate the speed of sound.

            'ISO':
            The default is 'ISO'.
        """
        # plot the speed of sound and the attenuation factor for each
        # measurement

        pass
