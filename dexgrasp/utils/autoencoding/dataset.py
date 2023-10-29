import os
import torch
import numpy as np
from plyfile import PlyData
from torch.utils.data import Dataset
from scipy.stats import ortho_group
import glob

import numpy as np
import torch
import os
import os.path as osp
import open3d as o3d
import time
import yaml

class PointCloudsDex(Dataset):

    def __init__(self, dataset_path, labels, is_training=False, overfit = False):
        """
        Arguments:
            is_training: a boolean.
        """

        cfg_path = '/data2/haoran/3DGeneration/3DAutoEncoder/data/name_3165.yaml'
        with open(cfg_path, 'r') as f:
            cfg = yaml.safe_load(f)
        object_scale_dict = cfg['env']['object_code_dict']
        object_code_list = list(object_scale_dict.keys())

        mesh_root = "/data2/haoran/3DGeneration/3DAutoEncoder/data/meshdatav3_pc_fps_new"
        dataset_root_path = "/data2/haoran/3DGeneration/3DAutoEncoder/data/UniDexGrasp/posedata"
        pose_root = "/data2/haoran/3DGeneration/3DAutoEncoder/data/fea.npy"
        self.data_dict = np.load(pose_root, allow_pickle=True)
        # import pdb
        # pdb.set_trace()
        num_pts = 1024 
        scale2str = {
            0.06: '006',
            0.08: '008',
            0.10: '010',
            0.12: '012',
            0.15: '015',
        }
        self.paths = []
        index = 0
        for object_code in object_code_list:
            # object_code = "sem/Bottle-8309e710832c07f91082f2ea630bf69e"
            mesh_dir = osp.join(mesh_root, object_code)
            dataset_path = dataset_root_path + '/' + object_code
            data_num_list = os.listdir(dataset_path)
            l_max = len(data_num_list)
            for num_i in range(min(100, l_max)):
                # num = data_num_list[num_i]
                # data_dict = dict(np.load(os.path.join(dataset_path, num), allow_pickle=True))
                # scale_inverse = data_dict['scale'].item()  # the inverse of the object's scale
                # scale = round(1 / scale_inverse, 2)
                # scale_str = scale2str[scale]
                

                npy_dir = osp.join(mesh_dir, f'coacd/pc_fps{num_pts}_010.npy')
                self.paths.append((npy_dir, index))
                index += 1

        # import pdb
        # pdb.set_trace()
        MAX_NUM = 20000
        self.is_training = is_training
        # self.paths = paths
        # if is_training == False:
        # if overfit:
        #     self.paths = self.paths[:5]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        """
        Returns:
            x: a float tensor with shape [3, num_points].
        """
        
        npy_dir, index = self.paths[i]
        
        object_euler_xy = self.data_dict[index][-3:-1]
        object_euler_z = self.data_dict[index][-1]
        
        with open(npy_dir, 'rb') as f:
            pts = np.load(f)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts[:, :3])
        # import pdb
        # pdb.set_trace()
        # print((object_euler_xy[0], object_euler_xy[1], object_euler_z[0]))
        R = pcd.get_rotation_matrix_from_xyz((object_euler_xy[0], object_euler_xy[1], object_euler_z))
        pcd.rotate(R, center=(0, 0, 0))
        # import pdb
        # pdb.set_trace()
        x = np.asarray(pcd.points)
        
        x -= x.mean(0)
        d = np.sqrt((x ** 2).sum(1))
        x /= d.max()

        if self.is_training:
            x = augmentation(x)
        
        x = torch.FloatTensor(x).permute(1, 0)
        return x



class PointCloudsDexMy(Dataset):

    def __init__(self, dataset_path, labels, is_training=False, overfit = False):
        """
        Arguments:
            is_training: a boolean.
        """

        cfg_path = '/data2/haoran/3DGeneration/3DAutoEncoder/data/name_3165.yaml'
        with open(cfg_path, 'r') as f:
            cfg = yaml.safe_load(f)
        object_scale_dict = cfg['env']['object_code_dict']
        object_code_list = list(object_scale_dict.keys())

        mesh_root = "/data2/haoran/3DGeneration/3DAutoEncoder/data/meshdatav3_pc_fps_new"
        dataset_root_path = "/data2/haoran/3DGeneration/3DAutoEncoder/data/UniDexGrasp/posedata"
        pose_root = "/data2/haoran/3DGeneration/3DAutoEncoder/data/info_dict.npy"
        
        # import pdb
        # pdb.set_trace()
        num_pts = 1024 
        scale2str = {
            0.06: '006',
            0.08: '008',
            0.10: '010',
            0.12: '012',
            0.15: '015',
        }
        self.paths = []
        index = 0
        info_dict = np.load(pose_root,allow_pickle=True).item()
        self.data_dict = info_dict
        for data_i in info_dict:
            data = info_dict[data_i]
            object_code = data["code"]
            scale = data["scale"]
            feat = data["feat"]
            mesh_dir = osp.join(mesh_root, object_code)
            npy_dir = osp.join(mesh_dir, f'coacd/pc_fps{num_pts}_{scale2str[scale]}.npy')
            self.paths.append((npy_dir, index))
            index += 1
        #     import pdb
        #     pdb.set_trace()
        # for object_code in object_code_list:
        #     # object_code = "sem/Bottle-8309e710832c07f91082f2ea630bf69e"
        #     mesh_dir = osp.join(mesh_root, object_code)
        #     dataset_path = dataset_root_path + '/' + object_code
        #     data_num_list = os.listdir(dataset_path)
        #     l_max = len(data_num_list)
        #     for num_i in range(min(100, l_max)):
        #         # num = data_num_list[num_i]
        #         # data_dict = dict(np.load(os.path.join(dataset_path, num), allow_pickle=True))
        #         # scale_inverse = data_dict['scale'].item()  # the inverse of the object's scale
        #         # scale = round(1 / scale_inverse, 2)
        #         # scale_str = scale2str[scale]
                

        #         npy_dir = osp.join(mesh_dir, f'coacd/pc_fps{num_pts}_010.npy')
        #         self.paths.append((npy_dir, index))
        #         index += 1

        # import pdb
        # pdb.set_trace()
        MAX_NUM = 20000
        self.is_training = is_training
        # self.paths = paths
        # if is_training == False:
        # if overfit:
        #     self.paths = self.paths[:5]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        """
        Returns:
            x: a float tensor with shape [3, num_points].
        """
        
        npy_dir, index = self.paths[i]
        # import pdb
        # pdb.set_trace()
        
        object_euler_xy = self.data_dict[index]["feat"][-3:-1]
        object_euler_z = self.data_dict[index]["feat"][-1]
        
        with open(npy_dir, 'rb') as f:
            pts = np.load(f)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts[:, :3])
        # import pdb
        # pdb.set_trace()
        # print((object_euler_xy[0], object_euler_xy[1], object_euler_z[0]))
        R = pcd.get_rotation_matrix_from_xyz((object_euler_xy[0], object_euler_xy[1], object_euler_z))
        pcd.rotate(R, center=(0, 0, 0))
        # import pdb
        # pdb.set_trace()
        x = np.asarray(pcd.points)
        
        x -= x.mean(0)
        d = np.sqrt((x ** 2).sum(1))
        x /= d.max()

        if self.is_training:
            x = augmentation(x)
        
        x = torch.FloatTensor(x).permute(1, 0)
        return x


class PointClouds(Dataset):

    def __init__(self, dataset_path, labels, is_training=False):
        """
        Arguments:
            is_training: a boolean.
        """

        paths = []
        for path, subdirs, files in os.walk(dataset_path):
            for name in files:
                p = os.path.join(path, name)
                assert p.endswith('.ply')
                paths.append(p)
        
        def get_label(p):
            return p.split('/')[-2]
        
        paths = [p for p in paths if get_label(p) in labels]
        self.is_training = is_training
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        """
        Returns:
            x: a float tensor with shape [3, num_points].
        """
        
        p = self.paths[i]
        x = load_ply(p)
        
        x -= x.mean(0)
        d = np.sqrt((x ** 2).sum(1))
        x /= d.max()

        if self.is_training:
            x = augmentation(x)
        
        x = torch.FloatTensor(x).permute(1, 0)
        return x


def load_ply(filename):
    """
    Arguments:
        filename: a string.
    Returns:
        a float numpy array with shape [num_points, 3].
    """
    ply_data = PlyData.read(filename)
    points = ply_data['vertex']
    points = np.vstack([points['x'], points['y'], points['z']]).T
    return points.astype('float32')


from scipy.stats import ortho_group

def augmentation(x):
    """
    Arguments:
        x: a float numpy array with shape [b, n, 3].
    Returns:
        a float numpy array with shape [b, n, 3].
    """

    jitter = np.random.normal(0.0, 1e-2, size=x.shape)
    x += jitter.astype('float32')

    # batch size
    b = x.shape[0]

    # random rotation matrix
    m = ortho_group.rvs(3)  # shape [b, 3, 3]
    m = np.expand_dims(m, 0)  # shape [b, 1, 3, 3]
    m = m.astype('float32')

    x = np.expand_dims(x, 1)
    x = np.matmul(x, m)
    x = np.squeeze(x, 1)

    return x
