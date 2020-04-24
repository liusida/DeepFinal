"""
strategy: 
search in 6x resolution, so we can quickly get a good base morphology (limbs).
then upsampling to 30x, and evolve from there.
"""

import random
import numpy as np
from voxelyze.helper import cprint

DNN = True
seed = 1000
experiment_name = "exp"
def init_all(in_DNN=True, in_seed=1000):
    global DNN, seed, experiment_name
    DNN = in_DNN
    seed = in_seed
    random.seed(seed)
    np.random.seed(seed)
    if DNN:
        wo = ''
    else:
        wo = 'wo_'
    experiment_name = f"Surrogate_{wo}DNN_{seed}"

# ============== should be able to change during evolution: ================
#  Plan: evolve a 100x100x100 body
#  target generation = 500
# 1. effect at this generation:
# only if fitness score pass new height multiple times, increase body dimension
# otherwise, keep adapting.

best_last_round = 0
body_dimension_n = 6
fitness_score_surpass_time = 0

def init_body_dimension_n(n):
    global body_dimension_n
    body_dimension_n = n

def body_dimension(generation=0, fitness_scores=[0]):
    # if generation<10:
    return [6,6,6]
    # else:
        # return [8,8,8]
        # return [30,30,30]

def mutation_rate(generation=0):
    # 19 times weight change, 1 time activation change
    ret = [19, 0.1]
    return ret

def target_population_size(generation=0):
    return 24

# =================== cannot change during evolution: =======================

hidden_layers = [10,10,10]
