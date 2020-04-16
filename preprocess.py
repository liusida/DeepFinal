import glob, os, shutil
import numpy as np
from lxml import etree


def load_data(report_filename, vxd_foldername):
    report = etree.parse(report_filename)
    robots = []
    distances = []
    for robot in range(9999):
        if len(report.xpath(f'//report//detail//robot_{robot:04}//initialCenterOfMass//x')) == 0:
            break
        init_x = float(report.xpath(f'//report//detail//robot_{robot:04}//initialCenterOfMass//x')[0].text)
        init_y = float(report.xpath(f'//report//detail//robot_{robot:04}//initialCenterOfMass//y')[0].text)
        end_x = float(report.xpath(f'//report//detail//robot_{robot:04}//currentCenterOfMass//x')[0].text)
        end_y = float(report.xpath(f'//report//detail//robot_{robot:04}//currentCenterOfMass//y')[0].text)
        distance = np.sqrt( (end_x-init_x)**2 + (end_y-init_y)**2 )

        # read file
        vxd_filename = f"{vxd_foldername}/robot_{robot:04}.vxd"
        vxd = etree.parse(vxd_filename)
        x = int(vxd.xpath('//VXD//Structure//X_Voxels')[0].text)
        y = int(vxd.xpath('//VXD//Structure//Y_Voxels')[0].text)
        z = int(vxd.xpath('//VXD//Structure//Z_Voxels')[0].text)
        assert x==6 and y==6 and z==6

        robot = np.zeros((x*y,z,5), dtype=np.float)
        layers = vxd.xpath('//VXD//Structure//Data//Layer')
        z=0
        for layer in layers:
            xy=0
            for c in layer.text:
                c = int(c)
                assert c>=0 and c<5
                robot[xy,z,c] = 1
                xy+=1
            z+=1
        distances.append([distance])
        robots.append(robot)

    robots = np.array(robots)
    distances = np.array(distances)
    return robots, distances

if __name__ == "__main__":
    # test
    report_filename = "data/BBTwo1000/generation_0001/report/output.xml"
    vxd_foldername = "data/BBTwo1000/generation_0001/start_population"
    robots, distances = load_data(report_filename, vxd_foldername)
    print(robots.shape)
    print(distances.shape)
