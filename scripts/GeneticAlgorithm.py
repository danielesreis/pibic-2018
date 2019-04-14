import pandas as pd
import numpy as np
import math
import random

class GeneticAlgorithm():

    def __init__(self, build_model, population_size, mutation_rate, iterations_number, max_var, variables, crossover_prob):
        self.build_model = build_model
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.iterations_number = iterations_number
        self.max_var = max_var
        self.variables = variables
        self.crossover_prob = crossover_prob

        self._cromossomes = None
        self._new_generation = None
        self._fitness_values = None
        random.seed(0)

    @property
    def new_generation(self):
        return self._new_generation

    @new_generation.setter
    def new_generation(self, new_generation):
        self._new_generation = new_generation

    @property
    def cromossomes(self):
        return self._cromossomes

    @cromossomes.setter
    def cromossomes(self, cromossomes):
        self._cromossomes = cromossomes

    @property
    def fitness_values(self):
        return self._fitness_values

    @fitness_values.setter
    def fitness_values(self, fitness_values):
        self._fitness_values = fitness_values
        
    # flow: 
    # generate a population with size equal to population_size (how to generate this pop? randomly? or from potentially good wavelengths?)
    # do the next steps til the stop criterion is met:
    # build models and compute the fitness function (explained variance for PLS maybe?)
    # do the next steps til the new generation has the same size as its predecessor:
    # select two cromossomes (how? randomly? tournament?)
    # reproduce them (how?)
    # mutation on the new generation
    # elitism maybe?

    def best_cromossome(self):
        print('Im assuming the best cromossome is the one with the highest fitness function!!!!!!')
        return self.fitness_values['fitness'].sort_values(ascending=False).iloc[0].index.value

    def initial_pop(self):
        cromossomes = pd.DataFrame(index=np.arange(self.population_size), columns=self.variables, data=0)
        for cromossome in range(self.population_size):
            rnd_wvl = random.sample(set(self.variables), self.max_var)
            cromossomes.iloc[cromossome][rnd_wvl] = 1
        return cromossomes

    def fit_model(self):
        fitness_values = np.array([0]*self.population_size)

        for cromossome in range(self.population_size):
            selected_wvls = self.variables[self.cromossomes.iloc[cromossome].values == 1]
            fitness_values[cromossome] = self.build_model(selected_wvls)

        self.fitness_values = pd.DataFrame(index=np.arange(self.population_size), columns=['cromossome', 'fitness'])
        self.fitness_values['cromossome'] = self.cromossomes
        self.fitness_values['fitness'] = fitness_values

    def selection(self):
        cromossomes = random.sample(set(np.arange(self.population_size)), 4)
        df = pd.DataFrame(index=np.arange(4), columns=['cromossome', 'fitness'])
        df['cromossome'] = cromossomes
        df['fitness'] = self.fitness_values[cromossomes]
        best_cromossomes = df['fitness'].sort_values(ascending=False).iloc[:2].index.values

        return best_cromossomes

    def reproduction(self, selected):
        prob = self.crossover_prob*100
        assert type(prob) is not int, 'A probabilidade do crossover deve ser inteira!'

        mask = pd.Series(np.array([0]*len(self.variables)))
        func = lambda x: int(not x) if random.randint(0, 99) < prob else x
        mask = mask.apply(func).values
        mask = int(''.join(str(val) for val in mask), 2)
        inv_mask = 2**len(self.variables)-1 - mask

        selected_1 = self.variables[selected[0]]
        selected_2 = self.variables[selected[1]]

        selected_1 = selected_1 & mask
        selected_2 = selected_2 & inv_mask
        child = selected_1 | selected_2
        child = [int(val) for val in bin(child)[2:]]
        return pd.Series([0]*(len(self.variables) - len(child)) + child)

    def mutation(self, cromossome):
        prob = self.mutation_rate*100
        assert type(prob) is not int, 'A taxa de mutacao deve ser inteira!'

        return cromossome.apply(lambda x: int(not x) if random.randint(0, 99) == prob else x)

    def increase_generation(self, index, cromossome):
        self.new_generation.iloc[index] = cromossome

    def otimize(self):
        self.cromossomes = self.initial_pop()
        print('should I be using RPD as the stop criteria? As I stated in the submission form?...')
        for __ in range(self.iterations_number):
            self.fit_model()
            for cromossome in range(self.population_size):
                selected = self.selection()
                new_cromossome = self.reproduction(selected)
                mutated = self.mutation(new_cromossome)
                self.increase_generation(cromossome, mutated)
            self.cromossomes = self.new_generation

        best_cromossome = self.best_cromossome()

        return self.variables[self.cromossomes.iloc[best_cromossome].values == 1]
