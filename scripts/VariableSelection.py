import numpy as np 
import math
import random

class VariableSelection():

    # def __init__(self, wavelengths):
        # possible_range_widths = [10, 20, 30, 40, 50]
        # possible_n_ranges = [1, 2, 3, 4, 5]
        # population_size = 10
        # wavelenghts = 
        # max_wv = max(wavelengths)
        # min_wv = min(wavelengths)
        # max_first_digit = 
        # MUTATION_PROB = 

    def random_pop(self):
        chromossomes = np.array([[0]*max(self.possible_n_ranges)*2]*self.population_size)
        for solution in range(self.population_size):
            selected_n_range = random.choice(self.possible_n_ranges)
            array_slice = math.floor(len(self.wavelengths)/selected_n_range)
            
            for rng in range(selected_n_range):
                slc = self.wavelengths[rng*array_slice:array_slice*(rng+1)]
                if rng == selected_n_range-1:
                    slc = self.wavelengths[rng*array_slice:]
                
                selected_range_width = random.choice(self.possible_range_widths)
                initial_wavelength = random.randint(slc[0], slc[-selected_range_width+1])

                chromossomes[solution][rng*2] = initial_wavelength
                chromossomes[solution][rng*2+1] = selected_range_width

        return chromossomes

    def random_selection(self):

    def reproduction(self):

    def mutation(self, chromossome):
    #     array com 0 nas últimas posições?
        new_chromossome = np.array([0]*max(self.possible_n_ranges)*2)
        for value, ctr in zip(chromossome, range(max(self.possible_n_ranges)*2)):
            while(True):
                new_num = []
                for i in str(value):
                    prob = random.randint(1,100)
                    if (ctr+1)%2 != 0 and ctr == 0:
                        value_to_append = random.randint(0,self.max_first_digit) if prob <= self.MUTATION_PROB*100 else int(i)
                        new_num.append(value_to_append)
                    else:
                        value_to_append = random.randint(0,9) if prob <= self.MUTATION_PROB*100 else int(i)
                        new_num.append(value_to_append)
                
                int_new_num = int(''.join(str(i) for i in new_num))
                if ((ctr+1)%2 != 0 and int_new_num > 0 int_new_num < self.max_wv):
                    new_chromossome[ctr] = int_new_num
                    break
                elif ((ctr+1)%2 == 0 and int_new_num > 0 and int_new_num+new_chromossome[ctr-1] < self.max_wv):
                    new_chromossome[ctr] = int_new_num
                    break
        
        return new_chromossome
