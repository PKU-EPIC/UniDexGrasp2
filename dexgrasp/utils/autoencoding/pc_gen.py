from os.path import join as pjoin

import numpy as np




def save_point_cloud_to_ply(points, colors, save_name='01.ply', save_root='/home/haorangeng/PointGroup_raw/dataset/visualization_self_space'):
    '''
    Save point cloud to ply file
    '''
    PLY_HEAD = f"ply\nformat ascii 1.0\nelement vertex {len(points)}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
    file_sting = PLY_HEAD
    for i in range(len(points)):
        file_sting += f'{points[i][0]} {points[i][1]} {points[i][2]} {int(colors[i][0])} {int(colors[i][1])} {int(colors[i][2])}\n'
    f = open(pjoin(save_root, save_name), 'w')
    f.write(file_sting)
    f.close()

pc = np.load("/data2/haoran/3DGeneration/3DAutoEncoder/0_31.npy")
save_point_cloud_to_ply(pc[0,:,:3], pc[0,:,3:6]*255, "0_31.ply", "/data2/haoran/3DGeneration/3DAutoEncoder")

pc = np.load("/data2/haoran/3DGeneration/3DAutoEncoder/2_80.npy")
save_point_cloud_to_ply(pc[0,:,:3], pc[0,:,3:6]*255, "2_80.ply", "/data2/haoran/3DGeneration/3DAutoEncoder")
