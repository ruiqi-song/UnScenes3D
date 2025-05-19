#!/usr/bin/env python3
# coding=utf-8
'''
   brief: 
   Author: dingbaiyong && baiyong.ding@waytous.com
   Date: 2024-09-20 16:27:52
   FilePath: /god_depth/tools/datasets/occupancy/genert_semi_pt.py
   Description: 
   LastEditTime: 2024-09-20 16:27:53
   LastEditors: dingbaiyong
   Copyright (c) 2024 by Inc, All Rights Reserved.
   '''

from manifast import *
from pipline.uns_label4d.base.database import Database
from pipline.uns_label4d.base.database import palette, class_to_name_seg


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


def maiin():
    db = Database('./data/raw_data',
                  sweep=False)

    for clip_idx in tqdm(range(0, len(db.clip_stamps.keys()))):
        save_path = None
        clip_name = list(db.clip_stamps.keys())[clip_idx]
        stamps = db.clip_stamps[clip_name]
        print(clip_name, stamps[0], stamps[-1])
        static_obs = db.load_label_4d(clip_name, stamps)
        static_cloud_map = db.build_semantic_map(clip_name)
        T_lidar_odom = db.load_calib(clip_name, stamps[0]).T_lidar_odom
        T_odom_lidar = np.linalg.inv(T_lidar_odom)
        obs_cloud, non_obs_cloud = db.build_static_obs_cloud(
            static_cloud_map, static_obs)
        wall_pc = non_obs_cloud[np.all(
            non_obs_cloud[:, 3:] == (128, 128, 128), axis=1)]
        road_pc = non_obs_cloud[np.all(
            non_obs_cloud[:, 3:] == (0, 128, 200), axis=1)]
        wall_pc = db.transform_pc(wall_pc, T_odom_lidar)
        road_pc = db.transform_pc(road_pc, T_odom_lidar)
        road_wall_pc_filter, road_bev_img, x_min, y_min, scale, img_size_x, img_size_y, bev_map = db.filter_connected_components4wall_road(
            wall_pc, road_pc)

        bev_map_path = os.path.join(
            db.data_dir, clip_name, f'bev_map/{stamps[0]}.png')
        make_path_dirs(bev_map_path)
        cv2.imwrite(bev_map_path, bev_map)
        bev_map_info_path = os.path.join(
            db.data_dir, clip_name, f'bev_map/{stamps[0]}_info.txt')
        infos_data = np.array([x_min, y_min, scale, img_size_x,
                               img_size_y]).astype(np.float32)
        np.savetxt(bev_map_info_path, infos_data)

        road_wall_pc_filter = db.transform_pc(
            road_wall_pc_filter, T_lidar_odom)
        semi_cloud_finetune = np.concatenate(
            [road_wall_pc_filter, obs_cloud])

        for stamp in tqdm(stamps):
            lidar_pc = db.load_lidar(clip_name, stamp)
            lidar_pc = np.hstack(
                (lidar_pc[:, :3], np.ones((lidar_pc.shape[0], 3))))
            t_lidar2odom = db.load_calib(clip_name, stamp).T_lidar_odom
            dynamic_obss = db.load_label3d(clip_name, stamp)
            dyna_obs_cloud_, non_dyna_obs_cloud_ = db.split_cloud_with_dyna_static(
                lidar_pc, dynamic_obss)
            lidar_pc_global = db.transform_pc(
                non_dyna_obs_cloud_, t_lidar2odom)
            obs_cloud_, non_obs_cloud_ = db.build_static_obs_cloud(
                lidar_pc_global, static_obs)
            road_wall_cloud = db.transform_pc(
                non_obs_cloud_, T_odom_lidar)
            wall_road_cloud = []
            non_wall_road_cloud = []
            for pt in road_wall_cloud:
                x = int((pt[1] - x_min) / scale)
                y = img_size_y-int((pt[0] - y_min) / scale)
                if 0 <= x < img_size_x and 0 <= y < img_size_y:
                    if road_bev_img[x, y] == 255:
                        pt[3:] = (128, 128, 128)
                        wall_road_cloud.append(pt)
                    elif road_bev_img[x, y] == 128:
                        pt[3:] = (0, 128, 200)
                        wall_road_cloud.append(pt)
                    else:
                        pt[3:] = (255, 255, 255)
                        non_wall_road_cloud.append(pt)
                else:
                    pt[3:] = (255, 255, 255)
                    non_wall_road_cloud.append(pt)
            road_wall_pc_ = db.transform_pc(
                np.array(wall_road_cloud), T_lidar_odom)
            road_wall_pc_ = db.transform_pc(
                road_wall_pc_, np.linalg.inv(t_lidar2odom))
            non_wall_road_cloud_ = db.transform_pc(
                np.array(non_wall_road_cloud), T_lidar_odom)
            non_wall_road_cloud_ = db.transform_pc(
                non_wall_road_cloud_, np.linalg.inv(t_lidar2odom))
            obs_cloud_ = db.transform_pc(
                obs_cloud_, np.linalg.inv(t_lidar2odom))
            semi_cloud = np.concatenate(
                [road_wall_pc_, non_wall_road_cloud_, obs_cloud_, dyna_obs_cloud_], axis=0)
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
            pc_save_path = osp.join(
                db.data_dir, clip_name, f"pc_seman/{stamp}.npy")
            make_path_dirs(pc_save_path)
            np.save(pc_save_path, obs_cloud_out)

            bbox_save_path = osp.join(
                db.data_dir, clip_name, f"pc_bbox/{stamp}.npy")
            make_path_dirs(bbox_save_path)

            if len(dynamic_obss) < 1:
                continue
            bboxes = []
            for obs in dynamic_obss:
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


if __name__ == "__main__":
    maiin()
