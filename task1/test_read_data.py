import h5py
import numpy as np
import open3d as o3d
import argparse
from create_ring import create_ring_obj
parser = argparse.ArgumentParser()
parser.add_argument("data_path",type=str,help="dataset path")
args = parser.parse_args()

with h5py.File(args.data_path,"r") as hf:
    X = hf["X"][20]
    y = hf["y"][20]
    r1 = y[0]
    r2 = y[1]
    theta = y[2]
    #torus angle should not be too small
    phi = y[3]
    center = [y[4],y[5],y[6]]
    normal = [y[7],y[8]]

    mesh = create_ring_obj(r1,r2,theta,phi,center,normal)
    mesh.compute_triangle_normals()

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(X))
    o3d.visualization.draw([pcd,mesh])


