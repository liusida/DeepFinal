# Step 1. Generate data and simulated to get labels.

import voxelyze as vx
from voxelyze.evolution.cppn_alife.CPPNEvolution import CPPNEvolution
import os, time, shutil, sys

population_size = 2
if len(sys.argv)==2:
    population_size = int(sys.argv[1])

for i in range(100):
    version = time.strftime("%Y%m%d-%H%M%S")
    os.mkdir(f"farm/{version}")
    shutil.copy("assets/base.vxa", f"farm/{version}/base.vxa")
    e = CPPNEvolution([6,6,6], population_size, [0,0])
    e.init_geno(hidden_layers=[10,10,10])
    e.express()
    population = e.dump_dic()["population"]
    for robot_id in population:
        population[robot_id]["genotype"] = {}
        population[robot_id]["phenotype"]["phaseoffset"] = None
        vx.write_a_vxd(f"farm/{version}/robot_{robot_id:04}.vxd", population[robot_id])
    # simulating
    os.system(f"bin/Voxelyze3 -w bin/vx3_node_worker -i farm/{version} -o farm/{version}/report.xml -f")

# zip for download
os.system(f"zip farm/{version}.zip farm/{version} -r")
