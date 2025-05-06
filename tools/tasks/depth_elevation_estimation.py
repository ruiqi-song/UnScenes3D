#!/usr/bin/env python3
# coding=utf-8
'''
   brief: 
   Version: v1.0.0
   Author: Baiyong Ding  && baiyong.ding@waytous.com
   Date: 2024-11-20 17:54:47
   Description: 
   LastEditors: Baiyong Ding
   LastEditTime: 2024-11-20 17:57:58
   FilePath: /Uni3DScenes/tools/tasks/depth_elevation_est.py
   Copyright 2024  by Inc, All Rights Reserved. 
   2024-11-20 17:54:47
'''
import os
import json
from tqdm import tqdm
import os.path as osp
import shutil
import numpy as np


class Database:
    def __init__(self, data_dir, sweep=False):
        self.data_dir = data_dir
        self._load_timestamps(sweep)

    def _load_timestamps(self, sweep):
        clip_info_path = os.path.join(
            self.data_dir, 'imagesets/sample_sweep_info.json')
        with open(clip_info_path, "r", encoding='utf-8') as f:
            json_data = json.load(f)
        self.clip_info = {}
        for clip, value in json_data.items():
            frames = []
            frames.extend(value['samples'])
            if sweep:
                frames.extend(value['sweeps'])
            frames = sorted(frames)
            self.clip_info[clip] = frames


splits = {
    "train":
    ['0003', '0004', '0005', '0007', '0009', '0011', '0012', '0017', '0018', '0020', '0021', '0022', '0024', '0025', '0026', '0027', '0028', '0029', '0030', '0032', '0033', '0037', '0038', '0039', '0040', '0041', '0042', '0045', '0047', '0048', '0049', '0050', '0051', '0052', '0053', '0055', '0057', '0059', '0060', '0062', '0063', '0064', '0065', '0066', '0067', '0069', '0072', '0073', '0074', '0075',
     '0076', '0077', '0078', '0079', '0081', '0083', '0084', '0085', '0086', '0089', '0090', '0091', '0092', '0093', '0094', '0096', '0097', '0098', '0099', '0100', '0101', '0102', '0103', '0104', '0105', '0106', '0107', '0108', '0110', '0111', '0112', '0113', '0114', '0115', '0116', '0117', '0118', '0119', '0120', '0121', '0122', '0123', '0124', '0126', '0127', '0128', '0130', '0131', '0132', '0133'],
    "val":
    ['0000', '0008', '0014', '0016', '0019', '0023', '0031', '0034', '0035', '0043', '0044', '0046', '0054',
     '0056', '0058', '0061', '0068', '0070', '0082', '0087', '0088', '0109', '0125', '0129', '0134'],
    "test":  ['0000', '0008', '0014', '0016', '0019', '0023', '0031', '0034', '0035', '0043', '0044', '0046', '0054', '0056', '0058', '0061', '0068', '0070', '0082', '0087', '0088', '0109', '0125', '0129', '0134'],
}
if __name__ == "__main__":
    db = Database('./data', sweep=False)
    dataset_save_dir = './datasets/depth_elevation_task'
    imagesets = osp.join(dataset_save_dir, 'ImageSets')
    os.makedirs(imagesets, exist_ok=True)
    for k, v in tqdm(splits.items()):
        with open(os.path.join(imagesets, f"{k}.txt"), "w") as f:
            f.write("")
        for clip in v:
            stamps = sorted(db.clip_info[f'clip_{clip}'])
            for stamp in tqdm(stamps):
                img_path = os.path.join(
                    db.data_dir, f'images/{stamp}.jpg')
                depth_path = os.path.join(
                    db.data_dir, f'depths/{stamp}.png')
                height_path = os.path.join(
                    db.data_dir, f'elevation/{stamp}.png')
                if not osp.exists(depth_path) or not osp.exists(img_path) or not osp.exists(height_path):
                    print(depth_path, img_path, height_path)
                    continue

                img_name = osp.basename(img_path)[:-4]
                with open(os.path.join(imagesets, f"{k}.txt"), "a") as f:
                    f.write(img_name+'\n')
                img_save_path = osp.join(
                    dataset_save_dir, 'images', img_name+'.jpg')
                os.makedirs(os.path.dirname(img_save_path), exist_ok=True)
                depth_save_path = osp.join(
                    dataset_save_dir, 'depths', img_name+'.png')
                os.makedirs(os.path.dirname(depth_save_path), exist_ok=True)

                height_save_path = osp.join(
                    dataset_save_dir, 'heights', img_name+'.png')
                os.makedirs(os.path.dirname(height_save_path), exist_ok=True)

                shutil.copy(img_path, img_save_path)
                shutil.copy(depth_path, depth_save_path)
                shutil.copy(height_path, height_save_path)
