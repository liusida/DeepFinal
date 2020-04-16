import numpy as np
from ...helper import largest_component
from ..Evolution import Evolution
from .CPPN import CPPN

class CPPNEvolution(Evolution):
    def __init__(self, body_dimension=[1,1,1], target_population_size=1, mutation_rate=[1,0]):
        """
        mutation_rate is a list: 0->weight to fn ratio, 1->weight change rate
        """
        super(CPPNEvolution, self).__init__(body_dimension, target_population_size, mutation_rate)
        self.genotype_keys.append("CPPN")

    def init_geno(self, hidden_layers=[1]):
        super(CPPNEvolution, self).init_geno()
        for g in self.population["genotype"]:
            g["CPPN"] = CPPN()
            g["CPPN"].init(hidden_layers=hidden_layers, weight_mutation_std=self.mutation_rate[1])

    def express(self):
        self.population["phenotype"] = []

        for robot_id in range(len(self.population["genotype"])):
            # do resolution, no need. # body_float, phaseoffset, bone_proportion = self.population["genotype"][robot_id]["CPPN"].get_output(self.body_dimension)
            output_of_network = self.population["genotype"][robot_id]["CPPN"].get_output(self.body_dimension)
            """
            for ALife paper, use (0: empty, 1: passiveSoft, 2: passiveHard, 3: active+, 4:active-)
            shape: 0 or other
            muscleOrTissue: 1/2 or 3/4
            tissueType: 1 or 2
            muscleType: 3 or 4
            phaseoffset: not used.
            """
            """output.keys() = ['body', 'shape', 'muscleOrTissue', 'tissueType', 'muscleType', 'phaseoffset']"""
            # do resolution, no need. # bone_proportion = np.mean(bone_proportion)
            # get body integer value from float output, and zero out phaseoffset for non-voxel.
            body = np.zeros(self.body_dimension, dtype=int)

            body_proportion = 0.2 # force the body to be 80% full to avoid cube.
            threshold = np.quantile(output_of_network["shape"], body_proportion) 
            
            passive_soft = np.logical_and(np.logical_and(output_of_network["shape"] > threshold, output_of_network["muscleOrTissue"] <= 0.5),output_of_network["muscleType"]>0.5)
            body[passive_soft] = 1
            passive_hard = np.logical_and(np.logical_and(output_of_network["shape"] > threshold, output_of_network["muscleOrTissue"] <= 0.5),output_of_network["muscleType"]<=0.5)
            body[passive_hard] = 2
            active_plus = np.logical_and(np.logical_and(output_of_network["shape"] > threshold, output_of_network["muscleOrTissue"] > 0.5),output_of_network["muscleType"]>0.5)
            body[active_plus] = 3
            active_minus = np.logical_and(np.logical_and(output_of_network["shape"] > threshold, output_of_network["muscleOrTissue"] > 0.5),output_of_network["muscleType"]<=0.5)
            body[active_minus] = 4
            # phaseoffset not used
            phaseoffset = np.zeros_like(body)

            body = largest_component(body)
            # to make output phaseoffset cleaner:
            phaseoffset[body==0] = 0. # zero out empty voxels' phaseoffset
            phaseoffset[body==2] = 0. # zero out bones' phaseoffset
            self.population["phenotype"].append( {
                "body": body,
                "phaseoffset": phaseoffset,
            })

    def mutate_single_value(self, key, value):
        if key=="CPPN":
            ret = value.clone()
            ret.mutate(num_random_activation_functions=1, num_random_weight_changes=self.mutation_rate[0])
            return ret
        else:
            return value

    def load_dic(self, Evolution_dic):
        super(CPPNEvolution, self).load_dic(Evolution_dic)
        self.mutation_rate = Evolution_dic["mutation_rate"]
        for i in range(len(self.population["genotype"])):
            s = self.population["genotype"][i]["CPPN"]
            self.population["genotype"][i]["CPPN"] = CPPN()
            self.population["genotype"][i]["CPPN"].loads(s)
            self.population["genotype"][i]["CPPN"].weight_mutation_std = self.mutation_rate[1]
        self.express()