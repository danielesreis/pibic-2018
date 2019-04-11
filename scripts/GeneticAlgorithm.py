import numpy as np
import math
import random

class GeneticAlgorithm():

    def __init__(self, population_size, mutation_rate, iterations, runs, max_var):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.iterations = iterations
        self.runs = runs
        self.max_var = max_var
        
    # flow: 
    # generate a population with size equal to population_size (how to generate this pop? randomly? or from potentially good wavelengths?)
    # do the next steps til the stop criterion is met:
    # build models and compute the fitness function (explained variance for PLS maybe?)
    # do the next steps til the new generation has the same size as its predecessor:
    # select two cromossomes (how? randomly? tournament?)
    # reproduce them (how?)
    # mutation on the new generation
    # elitism maybe?