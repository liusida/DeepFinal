import lxml.etree as etree
import re, os, sys
import voxelyze as vx
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import numpy as np
def body_dimension(generation=0, fitness_scores=[0]):
    return [6,6,6]

def mutation_rate(generation=0):
    # 19 times weight change, 1 time activation change
    ret = [19, 0.1]
    return ret

def target_population_size(generation=0):
    return 240
experiment_name = sys.argv[1]
# experiment_name = "v040711"
max_gen = 101
if len(sys.argv)>2:
    max_gen = int(sys.argv[2])

control_exp=False
if "wo_" in experiment_name:
    control_exp = True    
x = []
x1 = []
x2 = []
body = []
mut = []
pop = []
bestsofar = []
for generation in range(max_gen):
    folder = vx.foldername_generation(experiment_name, generation)
    if folder is None:
        break
    report_filename = f"{folder}/report/output.xml"
    if not os.path.exists(report_filename):
        break
    report = etree.parse(report_filename)
    detail = report.xpath("/report/detail")[0]
    distances = []
    num_voxels = []
    end_zs = []
    # read all detail. robot_id and distance.
    for robot in detail:
        robot_id = int(re.search(r'\d+', robot.tag).group())
        init_x = float(robot.xpath("initialCenterOfMass/x")[0].text)
        init_y = float(robot.xpath("initialCenterOfMass/y")[0].text)
        init_z = float(robot.xpath("initialCenterOfMass/z")[0].text)
        end_x = float(robot.xpath("currentCenterOfMass/x")[0].text)
        end_y = float(robot.xpath("currentCenterOfMass/y")[0].text)
        end_z = float(robot.xpath("currentCenterOfMass/z")[0].text)
        distance = np.sqrt((end_x-init_x)**2 + (end_y-init_y)**2) / 0.1 / 1 # body length=0.1m, evaluate time=1sec
        num_voxel = int(robot.xpath("num_voxel")[0].text)
        distances.append(distance)
        end_zs.append(end_z)
        num_voxels.append(num_voxel)
    body.append(body_dimension(generation)[0])
    mut.append(mutation_rate(generation)[1])
    pop.append(target_population_size(generation))
    x.append(distances)
    x1.append(num_voxels)
    x2.append(end_zs)
    best_this_gen = np.max(x)
    bestsofar.append(best_this_gen)

large_f = max(x[-1])
ticks = []
stepsize = 10
for i in range(len(x)):
    if i%stepsize==0:
        ticks.append(i)
flierprops = dict(marker='o', markerfacecolor='#999999', markersize=2, linestyle='none', markeredgecolor="#eeeeee")
whiskerprops = dict(color="#999999")
boxprops = dict(color="#999999")
plt.figure(figsize=[6,6])
plt.boxplot(x, positions=list(range(len(x))), flierprops=flierprops, whiskerprops=whiskerprops, boxprops=boxprops)
plt.xticks(ticks, ticks)
plt.plot(list(range(len(x))), bestsofar, label="Best so far")
plt.legend(loc='upper left')
if not control_exp:
    plt.annotate('Start using DNN', xy=(41, 0), xytext=(45, 0.1),
            arrowprops=dict(facecolor='black', shrink=0.05, width=2, headwidth=5),
            )
plt.ylabel("Travel Speed (body lengths/s)")
plt.xlabel("Generation")
plt.tight_layout()
plt.savefig(f"boxplot_{experiment_name}.svg")
plt.close()


plt.figure(figsize=(9,6))
pboxplot = plt.boxplot(x1, showfliers=False)
plt.xticks(ticks, ticks)
for patch in pboxplot['boxes']:
    patch.set_color("#DDDDDD")
plt.savefig("boxplot_num_voxels.png")
plt.close()

plt.figure(figsize=(9,6))
pboxplot = plt.boxplot(x2, showfliers=False)
plt.xticks(ticks, ticks)
plt.xlabel("Generation")
plt.ylabel("Height of the Center of Mass in the end.")
for patch in pboxplot['boxes']:
    patch.set_color("#DDDDDD")
plt.savefig("boxplot_end_z.png")
plt.close()