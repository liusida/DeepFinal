import glob, os, shutil, hashlib
import numpy as np
from lxml import etree
import torch

from voxelyze.helper import mkdir_if_not_exist

cache_folder = "tmp_numpy_cache"
mkdir_if_not_exist(cache_folder)
def load_data(report_filename, vxd_foldername):
    md5_filename = hashlib.sha256(f"{report_filename}{vxd_foldername}".encode("UTF-8")).hexdigest()
    if os.path.exists(f"{cache_folder}/{md5_filename}.robots.npy") and os.path.exists(f"{cache_folder}/{md5_filename}.distances.npy") :
        #read cache
        print(f"Loading {vxd_foldername} from cache...")
        robots = np.load(f"{cache_folder}/{md5_filename}.robots.npy")
        distances = np.load(f"{cache_folder}/{md5_filename}.distances.npy")

    else:
        print(f"Loading {vxd_foldername} from raw files...")
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
        #write cache
        np.save(f"{cache_folder}/{md5_filename}.distances", distances)
        np.save(f"{cache_folder}/{md5_filename}.robots", robots)

    return robots, distances

def load_data_torch(dataset, dtype=torch.FloatTensor):
    X = None
    Y = None
    for data in dataset:
        x, y = load_data(report_filename=data[0], vxd_foldername=data[1])
        x = torch.from_numpy(x).float().type(dtype)
        y = torch.from_numpy(y).float().type(dtype)

        if X is None:
            X = x
            Y = y
        else:
            X = torch.cat((X,x),0)
            Y = torch.cat((Y,y),0)
    print('before',X.size())
    X = X.view(-1,6,6,6,5)
    print('after',X.size())
    return X,Y

if __name__ == "__main__":
    # test
    report_filename = "data/BBTwo1000/generation_0001/report/output.xml"
    vxd_foldername = "data/BBTwo1000/generation_0001/start_population"
    robots, distances = load_data(report_filename, vxd_foldername)
    print(robots.shape)
    print(distances.shape)
