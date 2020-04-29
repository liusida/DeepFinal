# Evolution with and without DNN surrogate

# This file contains high level logic.

# psuedocode of this file
#
# if last generation of population exist
#   read them in
# else
#   randomly generate a population
# send the population to simulation
# read the output report
# select high fitness population
# use report to train a model
# use the model to mutate high fitness population
# write the next generation of population with high fitnesses and mutants


import random
import numpy as np
import sys

if len(sys.argv)>=3:
    DNN = bool(int(sys.argv[1]))
    seed = int(sys.argv[2])
    if len(sys.argv)==3:
        distinction = True
    else:
        distinction = bool(int(sys.argv[3]))
    random.seed(seed)
    np.random.seed(seed)
    if DNN:
        wo = ''
    else:
        wo = 'wo_'
    if distinction:
        ds = '_ds'
    else:
        ds = ''
    experiment_name = f"Surrogate_{wo}DNN_{seed}{ds}"

    print(f"DNN={DNN}, seed={seed}, distinction={distinction}", flush=True)

    print(f"Experiment name: {experiment_name}.", flush=True)

else:
    print("Usage:\n\npython 8.evolution.py <DNN> <seed>\n")
    exit()


best_last_round = 0
body_dimension_n = 6
fitness_score_surpass_time = 0

def init_body_dimension_n(n):
    global body_dimension_n
    body_dimension_n = n

def body_dimension(generation=0, fitness_scores=[0]):
    return [6,6,6]

def mutation_rate(generation=0):
    # 1 time weight change, 1 time activation change; weight std 0.5.
    ret = [4, 0.3]
    return ret

def target_population_size(generation=0):
    return 240

hidden_layers = [10,10,10]

assert target_population_size()%3==0

""" == Settings for Deep Learning == """
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import time, sys, math
import numpy as np

# my packages
import preprocess
import datasets
import visualize
import networks

torch.manual_seed(seed)

GPU = False
if GPU:
    dtype = torch.cuda.FloatTensor
    device = 'GPU'
else:
    dtype = torch.FloatTensor
    device = 'RAM'
# net = networks.FC4()
# if GPU:
#     net.cuda()
# Directly use pre-trained model
net = torch.load(f"./models/FC4_0.model")
net.to(torch.device('cpu'))
training_epochs_per_generation = 100
criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=0.001)
batch_size = 512
train_X = None
train_Y = None
# train_X.shape [?, 6, 6, 6, 5]
# train_Y.shape [?, 1]

def body_one_hot(body, dim=6, num_classes=5):
    """ body is a numpy [?,6,6,6] array, with max number of num_classes-1, say 4. """
    if GPU:
        dlong = torch.cuda.LongTensor
        dfloat = torch.cuda.FloatTensor
    else:
        dlong = torch.LongTensor
        dfloat = torch.FloatTensor
    body_t = torch.from_numpy(body).type(dlong)
    batch_size = body_t.size()[0]
    body_t = body_t.view(batch_size,dim,dim,dim,1)
    # One hot encoding buffer that you create out of the loop and just keep reusing
    body_onehot = dfloat(batch_size, dim,dim,dim, num_classes).zero_()
    body_onehot.scatter_(4, body_t, 1)
    return body_onehot

""" == Settings for Evolution == """
import voxelyze as vx
from voxelyze.evolution.cppn_alife.CPPNEvolution import CPPNEvolution
import numpy as np
import shutil, random, os
generation = 0
critical_generation = 5

try:
    shutil.rmtree(f"data/experiment_{experiment_name}")
except:
    pass
# vx.clear_workspace()

# try to resume from last experiment
evolution_dic, generation = vx.load_last_generation(experiment_name)
# if failed, start from scratch
if evolution_dic is None:
    generation = 0
    evolution = CPPNEvolution(body_dimension(), target_population_size(), mutation_rate())
    evolution.init_geno(hidden_layers=hidden_layers)
    evolution.express()
else:
    # resize using new body_dimension
    evolution = CPPNEvolution()
    init_body_dimension_n(evolution_dic["body_dimension"][0])
    evolution.load_dic(evolution_dic)

# infinity evolutionary loop
while(True):
    # write vxa vxd
    foldername = vx.prepare_directories(experiment_name, generation)
    vx.copy_vxa(experiment_name, generation)
    vx.write_all_vxd(experiment_name, generation, evolution.dump_dic())

    # before simulation, do a prediction
    if DNN:
        current_population_size = len(evolution.population['phenotype'])
        current_X = []
        for i in range(current_population_size):
            current_X.append( evolution.population['phenotype'][i]['body'] )
        current_X = np.array(current_X)
        current_X_t = body_one_hot(current_X, dim=current_X.shape[1])
        Y_hat = net(current_X_t)
        sorted_id = torch.argsort(Y_hat, dim=0, descending=True).cpu().numpy().reshape(-1)
        print(f"Predicted sort: {sorted_id}")

    # start simulator
    print("simulating...")
    vx.start_simulator(experiment_name, generation)
    # read report
    sorted_result = vx.read_report(experiment_name, generation)
    print(f"Result from simulation: {sorted_result['id']}")
    # Insert DNN and train
    if DNN:
        current_population_size = len(evolution.population['phenotype'])
        current_X = []
        current_Y = []
        for i in range(current_population_size):
            robot_id = sorted_result['id'][i]
            current_X.append( evolution.population['phenotype'][robot_id]['body'] )
            current_Y.append( sorted_result['fitness'][i])
        current_X = np.array(current_X)
        current_Y = np.array(current_Y).reshape([-1,1])
        current_X_t = body_one_hot(current_X)
        current_Y_t = torch.from_numpy(current_Y).type(dtype)
        # add to training set
        if train_X is None:
            train_X = current_X_t
            train_Y = current_Y_t
        else:
            train_X = torch.cat([train_X, current_X_t])
            train_Y = torch.cat([train_Y, current_Y_t])

        # train
        total_number = train_X.size()[0]
        for epoch in range(training_epochs_per_generation):
            for i in range(math.ceil(total_number/batch_size)):
                batch_start = i*batch_size
                batch_end = (i+1)*batch_size if (i+1)*batch_size<total_number-1 else total_number-1
                # print(f"{batch_start} - {batch_end}")
                train_X_batch = train_X[batch_start:batch_end]
                train_Y_batch = train_Y[batch_start:batch_end]

                optimizer.zero_grad()   # zero the gradient buffers
                Y_hat = net(train_X_batch)
                train_loss = criterion(Y_hat, train_Y_batch)
                train_loss.backward()
                optimizer.step()    # Does the update
        Y_hat = net(current_X_t)
        test_loss = criterion(Y_hat, current_Y_t)
        msg = f"epoch: {epoch:05}; train_loss: {train_loss:.5f}; test_loss: {test_loss:.5f}. "
        print(msg)
            

    # report the fitness
    top_n = 3 #len(sorted_result['id'])
    msg = f"Experiment {experiment_name}, simulation for generation {generation} finished.\nThe top {top_n} bestfit fitness score of this generation are \n"
    for i in range(top_n):
        if i<len(sorted_result['id']):
            robot_id = sorted_result['id'][i]
            msg += f"{evolution.population['genotype'][robot_id]['firstname']} {evolution.population['genotype'][robot_id]['lastname']}'s fitness score: {sorted_result['fitness'][i]:.1e} \n"
    print(msg, flush=True)

    # record a brief history for the bestfit
    print("recording...")
    vx.record_bestfit_history(experiment_name, generation, robot_id=sorted_result["id"][0], stopsec=2)

    # vx.write_box_plot(experiment_name, generation, sorted_result)

    # reporting
    # import sida.slackbot.bot as bot
    # bot.send(msg, 1, "GUB0XS56E")

    # dynamical sceduling
    evolution.target_population_size = target_population_size(generation)
    evolution.body_dimension = body_dimension(generation, sorted_result["fitness"])
    evolution.mutation_rate = mutation_rate(generation)

    # write report png
    # os.system("python plot_reports.py > /dev/null")
    # next generation
    generation += 1
    if distinction and generation>critical_generation and generation%critical_generation==0:
        print("---------------------------> Distinction!! ")
        evolution.init_geno(hidden_layers=hidden_layers)
        evolution.express()
    else:            
        if DNN and generation>critical_generation:
            next_generation = evolution.next_generation_with_prediction(sorted_result, net, body_one_hot)
        else:
            next_generation = evolution.next_generation(sorted_result)
