import torch
import json
from torch.utils.data import DataLoader
from dataset import PointClouds, PointCloudsDex
from trainer import Trainer
import os
import numpy as np
from os.path import join as pjoin

RUN_TAG = "run03"
NUM_EPOCHS = 24000
BATCH_SIZE = 32
CKPT_ROOT = f'ckpt/{RUN_TAG}/'
DEVICE = torch.device('cuda:0')
TRAIN_PATH = ''
VAL_PATH = ''
train_labels = ['sem', "core"]
val_labels = ['sem', "core", "ddg", "mujoco"]
dataset_path = '/data2/haoran/3DGeneration/3DAutoEncoder/data/meshdatav3_pc_fps_new'
TRAIN_LOGS = f'logs/{RUN_TAG}.json'
overfit = False
if overfit:
    SAVE_INTERVAL = 100
else:
    SAVE_INTERVAL = 2

CKPT_PATH = f"ckpt/{RUN_TAG}/2799.pth"
VISU_ROOT = "/data2/haoran/3DGeneration/3DAutoEncoder/visu"
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

def train_and_evaluate():
    if overfit:
        BATCH_SIZE = 5
    else:
        BATCH_SIZE = 32

    train = PointCloudsDex(dataset_path, train_labels, is_training=True, overfit = overfit)

    val = PointCloudsDex(dataset_path, val_labels, is_training=False)


    train_loader = DataLoader(
        dataset=train, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        dataset=val, batch_size=1, shuffle=False,
        num_workers=1, pin_memory=True
    )

    num_steps = NUM_EPOCHS * (len(train) // BATCH_SIZE)
    model = Trainer(num_steps, DEVICE)
    # model.network.to(DEVICE)
    if CKPT_PATH is not None:
        model.load(CKPT_PATH)

    i = 0
    logs = []
    text = 'e: {0}, i: {1}, loss: {2:.3f}'

    for e in range(NUM_EPOCHS):

        model.network.train()
        for x in train_loader:

            x = x.to(DEVICE)
            loss = model.train_step(x)
            if overfit:
                loss, x_restored = model.evaluate(x)
                import pdb
                pdb.set_trace()
                
                os.makedirs(VISU_ROOT, exist_ok=True)
                points_raw = x[0].permute(1, 0).cpu().numpy()
                import numpy as np
                color = np.zeros_like(points_raw)
                points_new = x_restored[0].permute(1, 0).cpu().numpy()
                # import pdb
                # pdb.set_trace()
                save_point_cloud_to_ply(points_raw, color, f"{i}_raw.ply", VISU_ROOT)
                save_point_cloud_to_ply(points_new, color, f"{i}_new.ply", VISU_ROOT)

            i += 1
            log = text.format(e, i, loss)
            print(log)
            logs.append(loss)
        if not overfit:
            eval_losses = []
            model.network.eval()
            for x in val_loader:

                x = x.to(DEVICE)
                loss, x_restored = model.evaluate(x)
                eval_losses.append(loss)

            # eval_losses = {k: sum(d[k] for d in eval_losses)/len(eval_losses) for k in loss.keys()}
            # eval_losses.update({'type': 'eval'})
            print(eval_losses)
            logs.append(eval_losses)
        os.makedirs(CKPT_ROOT, exist_ok=True)
        if e % SAVE_INTERVAL == 0:
            model.save(f"{CKPT_ROOT}{e}.pth")
        with open(TRAIN_LOGS, 'w') as f:
            json.dump(logs, f)


train_and_evaluate()
