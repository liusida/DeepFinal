# All operations related to subfolders and specific files should be put in this file
# So the structure is managed here only
#
from .helper import *
import os, re, json
import numpy as np
import lxml.etree as etree
child = etree.SubElement

Voxelyze3 = "./Voxelyze3"
vx3_node_worker = "./vx3_node_worker"

def foldername_generation(experiment_name, generation):
    return f"data/experiment_{experiment_name}/generation_{generation:04}"

def clear_workspace():
    import shutil, os
    if os.path.exists("workspace"):
        shutil.rmtree("workspace")

def prepare_directories(experiment_name, generation):
    mkdir_if_not_exist("data")
    mkdir_if_not_exist(f"data/experiment_{experiment_name}")
    base_dir = foldername_generation(experiment_name, generation)
    mkdir_if_not_exist(base_dir)
    sub_dir = ["start_population", "report", "mutation_model", "bestfit"]
    for d in sub_dir:
        mkdir_if_not_exist(f"{base_dir}/{d}")
    return base_dir

def copy_vxa(experiment_name, generation):
    import shutil
    foldername = foldername_generation(experiment_name, generation)
    shutil.copy("assets/base.vxa", f"{foldername}/start_population/base.vxa")

def read_report(experiment_name, generation):
    import re
    report_filename = f"{foldername_generation(experiment_name, generation)}/report/output.xml"
    report = etree.parse(report_filename)
    detail = report.xpath("/report/detail")[0]
    result = {"id": [], "fitness": []}
    # read all detail. robot_id and fitness.
    for robot in detail:
        robot_id = int(re.search(r'\d+', robot.tag).group())
        # fitness = float(robot.xpath("fitness_score")[0].text)
        init_x = float(robot.xpath("initialCenterOfMass/x")[0].text)
        init_y = float(robot.xpath("initialCenterOfMass/y")[0].text)
        init_z = float(robot.xpath("initialCenterOfMass/z")[0].text)
        end_x = float(robot.xpath("currentCenterOfMass/x")[0].text)
        end_y = float(robot.xpath("currentCenterOfMass/y")[0].text)
        end_z = float(robot.xpath("currentCenterOfMass/z")[0].text)
        # Fitness Function: Fitness Score
        fitness = (end_z-init_z) * 10 + np.sqrt((end_x-init_x)**2 + (end_y-init_y)**2)
        result["id"].append(robot_id)
        result["fitness"].append(fitness)
    # sort by fitness desc
    sorted_result = {"id": [], "fitness": []}
    args = np.argsort(result["fitness"])[::-1]
    for idx in args:
        sorted_result["id"].append(result["id"][idx])
        sorted_result["fitness"].append(result["fitness"][idx])
    return sorted_result

def copy_and_add_recordset(src, dst, stepsize, stopsec):
    best_fit = etree.parse(src)
    vxd = best_fit.xpath("/VXD")[0]
    RecordStepSize = child(vxd, "RecordStepSize")
    RecordStepSize.set("replace", "VXA.Simulator.RecordHistory.RecordStepSize")
    RecordStepSize.text = f"{stepsize}"
    # mtCONST = child(vxd, "mtCONST")
    # mtCONST.set("replace", "VXA.Simulator.StopCondition.StopConditionFormula.mtSUB.mtCONST")
    # mtCONST.text = f"{stopsec}"
    with open(dst, "wb") as file:
        file.write(etree.tostring(best_fit))

def record_bestfit_history(experiment_name, generation, robot_id, stepsize=100, stopsec=1):
    import shutil
    foldername = foldername_generation(experiment_name, generation)
    history_foldername = f"{foldername}/bestfit/"
    # report_filename = f"{foldername}/report/output.xml"
    # for root, dirs, files in os.walk(history_foldername):
    #     for f in files:
    #         os.remove(os.path.join(root, f))
    # report = etree.parse(report_filename)
    # best_fit_filename = report.xpath("/report/bestfit/filename")[0].text
    #vxd
    copy_and_add_recordset(f"{foldername}/start_population/robot_{robot_id:04}.vxd", f"{history_foldername}/robot_{robot_id:04}.vxd", stepsize, stopsec)
    #vxa
    shutil.copy("assets/base.vxa", f"{history_foldername}/base.vxa")
    #run (for convenience, we use linux pipeline here. if you use windows, please modify accrodingly.)
    commandline = f"{Voxelyze3} -i {history_foldername} -w {vx3_node_worker} > {history_foldername}/bestfit.history"
    run_shell_command(commandline)

def start_simulator(experiment_name, generation):
    # pipe output to log files
    foldername = foldername_generation(experiment_name, generation)
    commandline = f"{Voxelyze3} -i {foldername}/start_population/ -w {vx3_node_worker} -o {foldername}/report/output.xml -lf >> logs/{experiment_name}.log"
    run_shell_command(commandline)

def load_last_generation(experiment_name):
    max_generation_number = -1
    max_genration_foldername = ""
    if os.path.exists(f"data/experiment_{experiment_name}/"):
        folders = os.listdir(f"data/experiment_{experiment_name}/")
        for folder in folders:
            g = re.findall("[0-9]+", folder)
            if len(g)>=1:
                g = int(g[0])
                if g>max_generation_number:
                    max_generation_number = g
                    max_genration_foldername = folder
    if max_generation_number==-1:
        # previous generation not found
        return None, 0

    mutation_filename = f"data/experiment_{experiment_name}/generation_{max_generation_number:04}/mutation.json"
    with open(mutation_filename, 'r', encoding="UTF-8") as f:
        mutation_dic = json.load(f)
    other_fields_initiated = False
    max_genration_foldername = f"data/experiment_{experiment_name}/{max_genration_foldername}/start_population/"
    population_dic = {}
    for filename in os.listdir(max_genration_foldername):
        if filename[-4:]==".vxd":
            robot_id = int(re.findall(r'\d+', filename)[0])
            robot = {"phenotype":{}, "genotype":{}}
            xRoot = etree.parse(f"{max_genration_foldername}/{filename}")
            # Load body and phaseoffset
            x = int(xRoot.xpath("/VXD/Structure/X_Voxels")[0].text)
            y = int(xRoot.xpath("/VXD/Structure/Y_Voxels")[0].text)
            z = int(xRoot.xpath("/VXD/Structure/Z_Voxels")[0].text)
            #Body
            Layers = xRoot.xpath("/VXD/Structure/Data/Layer")
            lines = []
            for layer in Layers:
                line = []
                for ch in layer.text:
                    line.append(int(ch))
                lines.append(line)
            lines = np.array(lines)
            robot["phenotype"]["body"] = lines.reshape([z,y,x])
            #PhaseOffset
            Layers = xRoot.xpath("/VXD/Structure/PhaseOffset/Layer")
            lines = []
            for layer in Layers:
                line = []
                for ch in layer.text.split(","):
                    line.append(float(ch))
                lines.append(line)
            lines = np.array(lines)
            robot["phenotype"]["phaseoffset"] = lines.reshape([z,y,x])
            #Load other fields
            # if not other_fields_initiated:
            #     other_fields_initiated = True
            #     other_fields = xRoot.xpath("/VXD/OtherFields")[0]
            #     for key in other_fields.getchildren():
            #         robot[key.tag] = []
            other_fields = xRoot.xpath("/VXD/Genotype")[0]
            for key in other_fields.getchildren():
                robot["genotype"][key.tag] = key.text

            population_dic[robot_id] = robot

    mutation_dic["population"] = population_dic
    return mutation_dic, max_generation_number

def write_a_vxd(vxd_filename, robot):
    # Main Structure and PhaseOffset
    body = robot["phenotype"]["body"]
    phaseoffset = robot["phenotype"]["phaseoffset"]
    Z,Y,X = body.shape
    
    xRoot = etree.Element("VXD")
    # shrink the size of a single voxel according to the dimension of body, so that the only change is the resolution.
    Lattice_Dim = child(xRoot, "Lattice_Dim")
    Lattice_Dim.set('replace', 'VXA.VXC.Lattice.Lattice_Dim')
    Lattice_Dim.text = str(0.1/Z)
    # shrink end

    xStructure = child(xRoot, "Structure")
    xStructure.set('replace', 'VXA.VXC.Structure')
    xStructure.set('Compression', 'ASCII_READABLE')
    child(xStructure, "X_Voxels").text = str(X)
    child(xStructure, "Y_Voxels").text = str(Y)
    child(xStructure, "Z_Voxels").text = str(Z)
    body_flatten = body.reshape([Z,-1])
    xData = child(xStructure, "Data")
    for i in range(body_flatten.shape[0]):
        layer = child(xData, "Layer")
        str_layer = "".join([str(c) for c in body_flatten[i]])
        layer.text = etree.CDATA(str_layer)
    if phaseoffset is not None:
        phaseoffset_flatten = phaseoffset.reshape([Z,-1])
        xPhaseOffset = child(xStructure, "PhaseOffset")
        for i in range(phaseoffset_flatten.shape[0]):
            layer = child(xPhaseOffset, "Layer")
            str_layer = ",".join([f"{c:.03f}" for c in phaseoffset_flatten[i]])
            layer.text = etree.CDATA(str_layer)
    # Save other fields as well
    xOtherFields = child(xRoot, "Genotype")
    for key in robot["genotype"]:
        child(xOtherFields, key).text = robot["genotype"][key]
    # Save to file
    with open(vxd_filename, 'wb') as file:
        file.write(etree.tostring(xRoot))

def write_all_vxd(experiment_name, generation, mutation_dic):
    population_dic = mutation_dic["population"]
    for robot_id in range(len(population_dic)):
        robot = population_dic[robot_id]
        vxd_filename = f"data/experiment_{experiment_name}/generation_{generation:04}/start_population/robot_{robot_id:04}.vxd"
        write_a_vxd(vxd_filename, robot)
    
    mutation_filename = f"data/experiment_{experiment_name}/generation_{generation:04}/mutation.json"
    mutation_dic["population"] = None
    with open(mutation_filename, 'w', encoding="UTF-8") as f:
        json.dump(mutation_dic, f)

def write_box_plot(experiment_name, generation, sorted_result):
    import matplotlib.pyplot as plt
    png_filename = f"{foldername_generation(experiment_name, generation)}/report/generation_{generation}.png"

    plt.boxplot(sorted_result["fitness"])
    plt.savefig(png_filename)
    plt.close()

if __name__ == "__main__":
    def test_write_vxd():
        import shutil
        import numpy as np
        body = np.zeros([3,3,3], dtype=int)
        body[0,1,1:] = 1
        body[0,0,0] = 1
        body[2,2,2] = 1
        mkdir_if_not_exist("tmp")
        write_vxd("tmp/1.vxd", body, body)
        shutil.rmtree("tmp")
    # test_write_vxd()