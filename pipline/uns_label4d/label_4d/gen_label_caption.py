#!/usr/bin/env python3
# coding=utf-8
"""
brief: 
Version: v0.0.1
Author: knightdby  && knightdby@163.com
Date: 2025-05-15 13:51:53
Description: 
LastEditors: knightdby
LastEditTime: 2025-05-19 17:08:43
FilePath: /UnScenes3D/pipline/uns_label4d/label_4d/gen_label_caption.py
Copyright 2025 by Inc, All Rights Reserved. 
2025-05-15 13:51:53
"""


from manifast import *
from pipline.uns_label4d.base.database import Database
number_mapping = {
    1: "one",
    2: "two",
    3: "three",
    4: "four",
    5: "five",
    6: "six",
    7: "seven",
    8: "eight",
    9: "nine",
    10: "ten"
}

if __name__ == "__main__":

    db = Database('./data/raw_data',
                  sweep=False)

    for clip_idx in tqdm(range(0, len(db.clip_stamps.keys()), 1)):
        clip_name = list(db.clip_stamps.keys())[clip_idx]
        stamps = db.clip_stamps[clip_name]
        for stamp in stamps:
            image_path = osp.join(
                db.data_dir, clip_name, 'camera_1', f'{stamp}.jpg')
            label_2d_path = osp.join(
                db.data_dir, clip_name, 'camera_1_label_2d', f'{stamp}.json')
            label_2d = read_json_data(label_2d_path)
            obs_veh_num_left = []
            obs_veh_num_right = []
            obs_veh_num_front = []
            space_caption = ''
            obs_veh_caption = ''
            min_area = 0
            for obs in label_2d['instance']:
                obs['obj_type'] = obs['obj_type'].lower()
                x, y, w, h = obs['bbox'][0], obs['bbox'][1], obs['bbox'][2], obs['bbox'][3]
                area = w*h
                if obs['obj_type'] in ['road'] and area > min_area:
                    if x > 0.55:
                        space_caption = 'There is free space on the right side of the area.'
                    elif x < 0.45:
                        space_caption = 'There is free space on the left side of the area.'
                    else:
                        space_caption = 'There is free space in front of the area.'
                    min_area = area
                if obs['obj_type'] in ['truck', 'widebody', 'car',
                                       'excavator', 'machinery', 'person', 'rut', 'rock', 'slit', 'puddle']:
                    if x > 0.55:
                        obs_veh_num_right.append(obs['obj_type'])
                    elif x < 0.45:
                        obs_veh_num_left.append(obs['obj_type'])
                    else:
                        obs_veh_num_front.append(obs['obj_type'])
            if len(obs_veh_num_left) > 0:
                obs_veh_caption = 'the left side of the area has '
                for obs in set(obs_veh_num_left):
                    if obs not in ['rut', 'rock', 'slit', 'puddle']:
                        if obs_veh_num_left.count(obs) in number_mapping:
                            scriber = number_mapping[obs_veh_num_left.count(
                                obs)]
                        else:
                            scriber = obs_veh_num_left.count(obs)
                        obs_veh_caption += f"{scriber} {obs}"

                        if obs_veh_num_left.count(obs) > 1:
                            obs_veh_caption += 's, '
                        else:
                            obs_veh_caption += ', '
                if 'rut' in set(obs_veh_num_left) or 'rock' in set(obs_veh_num_left) or 'warningsign' in set(obs_veh_num_left):
                    obs_veh_caption += f"some barriers,"
                if 'slit' in set(obs_veh_num_left) or 'puddle' in set(obs_veh_num_left):
                    obs_veh_caption += f"some muddy areas,"

            if len(obs_veh_num_right) > 0:
                if len(obs_veh_num_left) > 0:
                    obs_veh_caption = obs_veh_caption[:len(
                        obs_veh_caption) - 2] + '; '
                obs_veh_caption += 'the right side of the area has '
                for obs in set(obs_veh_num_right):
                    if obs not in ['rut', 'rock', 'slit', 'puddle']:
                        if obs_veh_num_right.count(obs) in number_mapping:
                            scriber = number_mapping[obs_veh_num_right.count(
                                obs)]
                        else:
                            scriber = obs_veh_num_right.count(obs)
                        obs_veh_caption += f"{scriber} {obs}"
                        if obs_veh_num_right.count(obs) > 1:
                            obs_veh_caption += 's, '
                        else:
                            obs_veh_caption += ', '
                if 'rut' in set(obs_veh_num_right) or 'rock' in set(obs_veh_num_right) or 'warningsign' in set(obs_veh_num_right):
                    obs_veh_caption += f"some barriers,"
                if 'slit' in set(obs_veh_num_right) or 'puddle' in set(obs_veh_num_right):
                    obs_veh_caption += f"some muddy areas,"
            if len(obs_veh_num_front) > 0:
                if len(obs_veh_num_left) > 0 or len(obs_veh_num_right) > 0:
                    obs_veh_caption = obs_veh_caption[:len(
                        obs_veh_caption) - 2] + ', and '
                obs_veh_caption += 'the front of the area has '
                for obs in set(obs_veh_num_front):
                    if obs not in ['rut', 'rock', 'slit', 'puddle']:
                        if obs_veh_num_front.count(obs) in number_mapping:
                            scriber = number_mapping[obs_veh_num_front.count(
                                obs)]
                        else:
                            scriber = obs_veh_num_front.count(obs)
                        obs_veh_caption += f"{scriber} {obs}"
                        if obs_veh_num_front.count(obs) > 1:
                            obs_veh_caption += 's, '
                        else:
                            obs_veh_caption += ', '
                if 'rut' in set(obs_veh_num_front) or 'rock' in set(obs_veh_num_front) or 'warningsign' in set(obs_veh_num_front):
                    obs_veh_caption += f"some barriers,"
                if 'slit' in set(obs_veh_num_front) or 'puddle' in set(obs_veh_num_front):
                    obs_veh_caption += f"some muddy areas,"
            if len(obs_veh_caption) > 0:
                obs_veh_caption = space_caption+'\n'+obs_veh_caption[:len(
                    obs_veh_caption) - 2].capitalize() + '.'
            else:
                obs_veh_caption = space_caption
            caption_save_path = os.path.join(
                db.data_dir, clip_name, f'label_caption/{stamp}.txt')
            make_path_dirs(caption_save_path)
            with open(caption_save_path, 'w') as f:
                f.write(obs_veh_caption)
