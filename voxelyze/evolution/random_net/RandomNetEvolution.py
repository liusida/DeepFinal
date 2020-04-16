# RandomNetMutation
# Use a random weighted neural network to expand the genotype string into a 3D phaseoffset parameter
# genotype: 32 byte string
#
from .. import helper
from .. import workflow
from .. import names as names
import numpy as np
import string
import random
import torch
import torch.nn as nn
import torch.nn.functional as F


class GeNet(nn.Module):
    def __init__(self):
        super(GeNet, self).__init__()
        self.fc1 = nn.Linear(32, 1000)
        self.conv1 = nn.ConvTranspose3d(
            1, 1, padding=1, dilation=1, kernel_size=3, output_padding=0)

    def forward(self, x):
        x = self.fc1(x)
        x = x.view(-1, 1, 10, 10, 10)
        x = self.conv1(x)
        return x

class RandomNetMutation:
    def __init__(self, body_dimension):
        """ Init Mutation with data """
        self.body_dimension = body_dimension
        self.network = GeNet()
        pass

    def get_population(self):
        """ Construct the population data structure
        body and phaseoffset is the phenotype;
        genotype is a string;
        firstname and lastname are strings;
        """
        return {"body": [], "phaseoffset": [], "genotype": [], "firstname": [], "lastname": []}

    def init_geno(self, population, target_population_size):
        """ random init, start from scratch """
        # random initialization
        for robot_id in range(target_population_size):
            # body_random = np.random.random(self.body_dimension)
            # body = np.zeros_like(body_random, dtype=int)
            # body[body_random < 0.5] = 1
            # body = largest_component(body)
            # phaseoffset = np.random.random(self.body_dimension)
            genotype = ''.join([random.choice(string.digits)
                                for n in range(32)])
            population["genotype"].append(genotype)
            # population["body"].append(body)
            # population["phaseoffset"].append(phaseoffset)
            population["firstname"].append(names.get_first_name())
            population["lastname"].append(names.get_last_name())

    def expression(self, population):
        """ express genotype (a list of digits) to phenotype (body and phaseoffset) """
        for robot_id in range(len(population["genotype"])):
            genotype = population["genotype"][robot_id]
            data_x = list(genotype)
            data_x = [int(x) for x in data_x]
            data_x = torch.tensor(data_x)
            data_x = data_x / 10.
            y = self.network(data_x)
            y = y.data.numpy().reshape(10, 10, 10)
            population["phaseoffset"].append(y)

            legs = list(
                "1110000111111000011111100001110000000000000000000000000000000000000000111000011111100001111110000111")
            legs = [int(x) for x in legs] * 4
            trunk = list(
                "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111")
            trunk = [int(x) for x in trunk] * 6
            body = np.array(legs + trunk).reshape(10,10,10)
            population["body"].append(body)


    def _mutate_geno(self, string):
        pos = random.randint(0, len(string)-1)
        string = list(string)
        string[pos] = str(( int(string[pos]) + 1 ) % 10)
        string = "".join(string)
        return string

    def mutate(self, population):
        """ Do mutation """
        from .. import names
        mutants = {}
        anykey = None
        for key in population.keys():
            mutants[key] = []
            anykey = key
        for i in range(len(population[anykey])):
            # copy all properties
            for key in population.keys():
                mutants[key].append(population[key][i])
            # change the first name
            if "firstname" in mutants:
                mutants["lastname"][-1] = mutants["firstname"][-1]
                mutants["firstname"][-1] = names.get_first_name()
            # change teh geno
            if "genotype" in mutants:
                mutants["genotype"][-1] = self._mutate_geno(mutants["genotype"][-1])

        return mutants
    
    def next_generation(self, sorted_result, population):
        # select the first half
        selected = workflow.empty_population_like(population)
        half = int(len(sorted_result["id"])/2)
        for i in sorted_result["id"][:half]:
            for key in population.keys():
                selected[key].append(population[key][i])

        # mutate first half to two mutant groups
        mutant1 = self.mutate(selected)
        mutant2 = self.mutate(selected)

        # combine two mutant groups into next generation
        next_generation = workflow.empty_population_like(population)
        workflow.add_population(mutant1, next_generation)
        workflow.add_population(mutant2, next_generation)
        return next_generation
