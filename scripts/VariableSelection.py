import numpy as np 
import math
import random

class VariableSelection():

    # def __init__(self):
        # possible_range_widths = [10, 20, 30, 40, 50]
        # possible_n_ranges = [1, 2, 3, 4, 5]
        # n_chromossomes = 10

    def random_pop(self, wavelengths):
        chromossomes = np.array([[0]*max(possible_n_ranges)*2]*n_chromossomes)
        for solution in range(n_chromossomes):
            selected_n_range = random.choice(possible_n_ranges)
            array_slice = math.floor(len(wavelengths)/selected_n_range)
            
            for rng in range(selected_n_range):
                slc = wavelengths[rng*array_slice:array_slice*(rng+1)]
                if rng == selected_n_range-1:
                    slc = wavelengths[rng*array_slice:]
                
                selected_range_width = random.choice(possible_range_widths)
                initial_wavelength = random.randint(slc[0], slc[-selected_range_width+1])

                chromossomes[solution][rng*2] = initial_wavelength
                chromossomes[solution][rng*2+1] = selected_range_width

        return chromossomes
