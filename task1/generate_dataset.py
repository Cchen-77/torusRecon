import h5py
import argparse
import numpy as np
from create_ring import create_ring_obj,sample_mesh_nearly

if __name__ == '__main__':
    data_size = 1000
    num_points = 1000
    
    X = np.zeros([data_size,num_points,3])
    y  = np.zeros([data_size,9])

    for i in range(data_size):
        r2 = np.random.uniform(5,20)
        r1 = r2 + np.random.uniform(10,100)
        theta = np.random.uniform(0,np.pi)
        #torus angle should not be too small
        phi = theta + np.random.uniform(np.pi/6,np.pi)
        normal = [np.random.uniform(0,np.pi),np.random.uniform(0,2*np.pi)]
        center = [np.random.uniform(-100,100),np.random.uniform(-100,100),np.random.uniform(-100,100)]

        compact_arg = [r1,r2,theta,phi] + center + normal
        y[i] = np.asarray(compact_arg)

        mesh = create_ring_obj(r1, r2, theta, phi, center, normal)
        pcd = sample_mesh_nearly(mesh,num_points)
        X[i] = np.asarray(pcd.points)

    with h5py.File('data.h5','w') as hf:
        hf.create_dataset("X",data=X)
        hf.create_dataset("y",data=y)




        








