#!/usr/bin/env python3
# coding=utf-8
"""
brief: 
Version: v0.0.1
Author: knightdby  && knightdby@163.com
Date: 2025-04-14 19:06:53
Description: 
LastEditors: knightdby
LastEditTime: 2025-05-14 14:20:06
FilePath: /UniOcc/uniocc/semantic/get_pc_semantic.py
Copyright 2025 by Inc, All Rights Reserved. 
2025-04-14 19:06:53
"""

import open3d as o3d
from manifast import *
from pipline.uns_label4d.base.database import Database, cloud_viewer
from shapely.geometry import Point, Polygon
from pipline.uns_label4d.base.database import palette, class_to_name_seg, name_to_class_seg, cloud_rgb_viewer_save, cloud_viewer, cloud_viewer_rgb_id, cloud_viewer_rgb

unstructured_road_name2classid = {'background': 0,
                                  'rock': 1, 'rut': 1,
                                  'truck': 2, 'widebody': 3, 'car': 4, 'excavator': 5, 'machinery': 6,
                                  'person': 7,
                                  'warningsign': 8, 'signboard': 8,
                                  #   'cable': 10,
                                  'puddle': 9, 'slit': 9,
                                  'building': 10, 'retainingwall': 10, 'fence': 10,
                                  'road': 11}
unstructured_road_classname = ['background', 'barrier',
                               'truck', 'widebody', 'car', 'excavator', 'machinery',
                               'pedestrian',
                               'traffic_cone',
                               'muddy',
                               'terrain',
                               'driveable_surface',
                               ]
unstructured_road_classid2name = {index: value for index,
                                  value in enumerate(unstructured_road_classname)}
unstructured_road_classname2id = {v: n for n,
                                  v in unstructured_road_classid2name.items()}

make_path_dirs('tlogs_exp/calib_check')
view = False
fail_scenes = ['scene_00071', 'scene_00280', 'scene_00295', 'scene_00297', 'scene_00298', 'scene_00299', 'scene_00301',
               'scene_00302', 'scene_00303', 'scene_00304', 'scene_00305', 'scene_00306', 'scene_00307', 'scene_00308',
               'scene_00309', 'scene_00310', 'scene_00311', 'scene_00312', 'scene_00313', 'scene_00314', 'scene_00319',
               'scene_00331', 'scene_00332', 'scene_00353', 'scene_00286', 'scene_00288', 'scene_00294', 'scene_00325',
               'scene_00329', 'scene_00330', 'scene_00340', 'scene_00413']


def maiin(dataset_dir):
    # view = True
    db = Database(dataset_dir, sweep=False)
    for clip_idx in tqdm(range(0, len(db.clip_stamps.keys()))):
        save_path = None
        clip_name = list(db.clip_stamps.keys())[clip_idx]
        stamps = db.clip_stamps[clip_name]
        if len(stamps) < 5:
            continue
        clip_stamp_min = stamps[0]
        clip_stamp_max = stamps[-1]
        map_points = None
        print(clip_name, clip_stamp_min, clip_stamp_max)
        if clip_name not in db.label_bevmaps:
            print("No bevmap: ", clip_name)
            continue
        # if clip_name not in fail_scenes:
        #     print("Skip scene: ", clip_name)
        #     continue
        static_obs = db.load_label4d(clip_name)
        static_invis_obs = db.load_label4d_invis(clip_name)
        bevmaps = db.load_bevmap(clip_name)
        prepared_polygons = db.prepare_bevmap_polygons(bevmaps)
        for stamp in tqdm(stamps):
            calib, _ = db.load_calib(stamp)
            lidar_pc = db.load_lidar(stamp)
            lidar_pc = lidar_pc[lidar_pc[:, 0] < 110, :]
            lidar_pc = np.hstack(
                (lidar_pc[:, :3], np.ones((lidar_pc.shape[0], 3))))
            dyna_obs = db.load_label3d(stamp)
            dyna_pc_, static_pc_ = db.split_dyna_static_cloud(
                lidar_pc, dyna_obs, view)
            static_pc_map_ = db.transform_pc(
                static_pc_, calib.T_lidar_odom)
            static_bbox_pc_, static_unkown_ovis_pc_ = db.split_static_bbox_cloud(
                static_pc_map_, static_obs)
            static_bbox_invis_pc_, static_unkown_pc_ = db.split_static_bbox_cloud(
                static_unkown_ovis_pc_, static_invis_obs, prepared_polygons)
            static_bavmap_pc_, unkown_pc_ = db.split_static_bevmap_cloud(
                static_unkown_pc_, prepared_polygons)
            if len(static_bavmap_pc_) == 0:
                static_semi_pc_map = np.array([[0, 0, 0, 0, 0, 0]])
            else:
                static_semi_pc_map = static_bavmap_pc_
            if len(static_bbox_pc_) > 0:
                static_semi_pc_map = np.concatenate(
                    [static_semi_pc_map, static_bbox_pc_], axis=0)
            if len(unkown_pc_) > 0:
                static_semi_pc_map = np.concatenate(
                    [static_semi_pc_map, unkown_pc_], axis=0)
            if len(static_bbox_invis_pc_) > 0:
                static_semi_pc_map = np.concatenate(
                    [static_semi_pc_map, static_bbox_invis_pc_], axis=0)
            if len(static_bavmap_pc_) == 0:
                static_semi_pc_map = static_semi_pc_map[1:, :]
            static_semi_pc = db.transform_pc(
                static_semi_pc_map, np.linalg.inv(calib.T_lidar_odom))
            semi_cloud = np.concatenate(
                [dyna_pc_, static_semi_pc], axis=0)
            obs_cloud_out = np.array([[0, 0, 0, 0, 0, 0, 0]])
            for i, palt in enumerate(palette):
                obs_cloud = semi_cloud[np.all(
                    semi_cloud[:, 3:] == list(palt), axis=1)]
                class_name = class_to_name_seg[i]
                if class_name not in unstructured_road_name2classid:
                    continue
                reset_idx = unstructured_road_name2classid[class_name]
                obs_cloud = np.hstack(
                    (obs_cloud[:, :3], np.ones((obs_cloud.shape[0], 1))*reset_idx, obs_cloud[:, 3:]))
                obs_cloud_out = np.concatenate(
                    [obs_cloud_out, obs_cloud],  axis=0)
            obs_cloud_out = obs_cloud_out[1:]

            pc_save_path = osp.join(
                dataset_dir, f"labels/pc_seman/{stamp}.npy")
            make_path_dirs(pc_save_path)
            np.save(pc_save_path, obs_cloud_out)
            bbox_save_path = osp.join(
                dataset_dir, f"labels/pc_bbox/{stamp}.npy")
            make_path_dirs(bbox_save_path)

            if len(dyna_obs) > 0:
                bboxes = []
                for obs in dyna_obs:
                    bbox = []
                    token = obs[0]
                    name = obs[1].lower()
                    if name not in unstructured_road_name2classid:
                        print('error class: ', name)
                        continue
                    bbox.extend([token, unstructured_road_name2classid[name]])
                    center = obs[2:5]
                    w, h, l, yaw = obs[5:]
                    bbox.extend(center)
                    bbox.extend([l, w, h, yaw])
                    bboxes.append(bbox)
                bboxes = np.array(bboxes)
                np.save(bbox_save_path, bboxes)
            if map_points is None:
                map_points = static_semi_pc_map
            else:
                map_points = np.concatenate(
                    (map_points, static_semi_pc_map), axis=0)

        map_points = db.transform_pc(
            map_points, np.linalg.inv(calib.T_lidar_odom))
        if map_points is not None:
            save_path = f'{dataset_dir}/scenes/semi_scenes/{clip_name}.png'
            cloud_rgb_viewer_save(map_points, save_path)
            if save_path:
                map_img = np.zeros((747, 640+320, 3))
                map_img_ = cv2.imread(save_path)
                img_1 = db.load_camera(stamps[0])
                img_1 = cv2.resize(img_1, (640, 480))
                img_2 = db.load_camera(stamps[int(len(stamps)/2)])
                img_2 = cv2.resize(img_2, (640, 480))
                img_3 = db.load_camera(stamps[-1])
                img_3 = cv2.resize(img_3, (640, 480))
                img = np.vstack((img_1, img_2, img_3))
                img = cv2.resize(img, (320, 747))
                map_img[:747, :320] = img
                map_img[:747, 320:] = map_img_
                cv2.imwrite(save_path, map_img)
        # break


if __name__ == "__main__":
    dataset_dir = '/media/knight/disk2knight/htmine_occ_2'
    maiin(dataset_dir)
