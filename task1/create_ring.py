"""
作者 ljx
创建一个管道的3D mesh模型文件 ring.obj
2024/4/7

mayble: 
pip install scipy
"""



import math
import numpy as np
from scipy.spatial.transform import Rotation as R
import open3d as o3d


# 将模型由normal1旋转到normal2
def rotate_model(vertices, normal1, normal2):
    # Normalize the normals
    normal1 = normal1 / np.linalg.norm(normal1)
    normal2 = normal2 / np.linalg.norm(normal2)
    # Compute the rotation axis and angle
    axis = np.cross(normal1, normal2)
    angle = math.acos(np.dot(normal1, normal2))
    # Create the rotation
    rotation = R.from_rotvec(axis * angle)
    # Apply the rotation to each vertex
    rotated_vertices = [rotation.apply(vertex) for vertex in vertices]
    return rotated_vertices

def create_ring_obj(r1, r2, theta, phi, center = [0,0,0], normal = [0, 0], alphaStep = 100, betaStep = 50):
    """
    转弯半径 r1
    环体半径 r2
    转弯起始 theta
    转弯终止 phi
    法向量 normal (使用球面坐标系表示 (theta,phi) theta in [0,pi] and phi in [0,2*pi])
    圆环中心 center
    alphaStep: alpha方向上的分割数
    betaStep: beta方向上的分割数

    遍历alpha, beta
        生成顶点坐标:
        x = (r1  + r2 * cos(beta)) * cos(alpha)
        y = (r1  + r2 * cos(beta)) * sin(alpha)
        z = r2 * sin(beta)
    """
    normal = [np.sin(normal[0])*np.cos(normal[1]),np.sin(normal[0])*np.sin(normal[1]),np.cos(normal[0])]  

    vertices = []
    faces = []
    alphas = np.linspace(theta, phi, alphaStep, endpoint=False)
    betas = np.linspace(0, 2*np.pi, betaStep, endpoint=False)
    
    # generate vertices
    for alpha in alphas:
        for beta in betas:
            x = (r1 + r2 * math.cos(beta)) * math.cos(alpha) 
            y = (r1 + r2 * math.cos(beta)) * math.sin(alpha) 
            z = r2 * math.sin(beta)
            vertices.append([x, y, z])
    # generate faces
    for i in range(alphaStep-1):
        for j in range(betaStep):
            v1 = i * betaStep + j
            v2 = i * betaStep + (j + 1) % betaStep
            v3 = ((i+1) % alphaStep) * betaStep + (j + 1) % betaStep
            v4 = ((i+1) % alphaStep) * betaStep + j
            faces.append([v1, v2, v3, v4])
    
    # rotate the model
    vertice2 = rotate_model(vertices, [0, 0, 1], normal)
    vertices = [[v[0] + center[0], v[1] + center[1], v[2] + center[2]] for v in vertice2]

    #for open3d interface
    triangles = []
    for face in faces:
        triangles.append([face[0],face[1],face[2]])
        triangles.append([face[2],face[3],face[0]])
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    
    #visualize
    #o3d.visualization.draw([mesh])
    #export obj
    #o3d.io.write_triangle_mesh("ring.obj", mesh)
    return mesh

def sample_mesh_nearly(mesh,num_points=1000,radius=0.1):
    mesh.compute_triangle_normals()
    points = []
    num_triangle_meshs = len(mesh.triangles)
    for _ in range(num_points):
        triangle_idx = np.random.randint(0,num_triangle_meshs)
        u = np.random.uniform(0,1)
        v = np.random.uniform(0,1)
        if u + v > 1:
            u = 1-u
            v = 1-v
        w = 1 - u -v
        point = u*mesh.vertices[mesh.triangles[triangle_idx][0]] + v*mesh.vertices[mesh.triangles[triangle_idx][1]] + w*mesh.vertices[mesh.triangles[triangle_idx][2]]
        point += np.random.uniform(-radius,radius)*mesh.triangle_normals[triangle_idx]
        points.append(point)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(points))
    return pcd

if __name__ == '__main__':
    mesh = create_ring_obj(100, 10, np.pi/4, 2*np.pi/3, [0.7, 0.3, 0.5], [0,0])
    pcd = sample_mesh_nearly(mesh,num_points=500)
    o3d.visualization.draw([pcd])

    