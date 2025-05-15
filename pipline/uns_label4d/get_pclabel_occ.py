#!/usr/bin/env python3
# coding=utf-8
"""
brief: 
Version: v0.0.1
Author: knightdby  && knightdby@163.com
Date: 2025-04-15 10:56:14
Description: 
LastEditors: knightdby
LastEditTime: 2025-05-15 18:17:01
FilePath: /UnScenes3D/pipline/uns_label4d/semantic/get_pclabel_occ.py
Copyright 2025 by Inc, All Rights Reserved. 
2025-04-15 10:56:14
"""


from manifast import *
from pipline.uns_label4d.base.database import Database, palette, name_to_class_seg, cloud_viewer, cloud_viewer_rgb_id
import yaml
import torch
from mmcv.ops.points_in_boxes import (points_in_boxes_all, points_in_boxes_cpu,
                                      points_in_boxes_part)
import open3d as o3d
from scipy.spatial.transform import Rotation
from copy import deepcopy
import chamfer


def run_poisson(pcd, depth, n_threads, min_density=None):
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=depth, n_threads=13
    )

    # Post-process the mesh
    if min_density:
        vertices_to_remove = densities < np.quantile(densities, min_density)
        mesh.remove_vertices_by_mask(vertices_to_remove)
    mesh.compute_vertex_normals()

    return mesh, densities


def create_mesh_from_map(buffer, depth, n_threads, min_density=None, point_cloud_original=None):

    if point_cloud_original is None:
        pcd = buffer_to_pointcloud(buffer)
    else:
        pcd = point_cloud_original

    return run_poisson(pcd, depth, n_threads, min_density)


def buffer_to_pointcloud(buffer, compute_normals=False):
    pcd = o3d.geometry.PointCloud()
    for cloud in buffer:
        pcd += cloud
    if compute_normals:
        pcd.estimate_normals()

    return pcd


def nn_correspondance(verts1, verts2):
    """ for each vertex in verts2 find the nearest vertex in verts1

        Args:
            nx3 np.array's
        Returns:
            ([indices], [distances])

    """
    import open3d as o3d

    indices = []
    distances = []
    if len(verts1) == 0 or len(verts2) == 0:
        return indices, distances

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(verts1)
    kdtree = o3d.geometry.KDTreeFlann(pcd)
    for vert in verts2:
        _, inds, dist = kdtree.search_knn_vector_3d(vert, 1)
        indices.append(inds[0])
        distances.append(np.sqrt(dist[0]))

    return indices, distances


def preprocess_cloud(
    pcd,
    max_nn=20,
    normals=None,
):

    cloud = deepcopy(pcd)
    if normals:
        params = o3d.geometry.KDTreeSearchParamKNN(max_nn)
        cloud.estimate_normals(params)
        cloud.orient_normals_towards_camera_location()

    return cloud


def preprocess(pcd, config):
    return preprocess_cloud(
        pcd,
        config['max_nn'],
        normals=True
    )


fail_scenes = ['scene_00071', 'scene_00280', 'scene_00295', 'scene_00297', 'scene_00298', 'scene_00299', 'scene_00301',
               'scene_00302', 'scene_00303', 'scene_00304', 'scene_00305', 'scene_00306', 'scene_00307', 'scene_00308',
               'scene_00309', 'scene_00310', 'scene_00311', 'scene_00312', 'scene_00313', 'scene_00314', 'scene_00319',
               'scene_00331', 'scene_00332', 'scene_00353', 'scene_00286', 'scene_00288', 'scene_00294', 'scene_00325',
               'scene_00329', 'scene_00330', 'scene_00340', 'scene_00413']


if __name__ == '__main__':
    from argparse import ArgumentParser
    parse = ArgumentParser()
    args = parse.parse_args()
    args.config_path = 'uniocc/base/config.yaml'
    args.with_semantic = True
    # args.whole_scene_to_mesh = True
    # args.to_mesh = False
    args.whole_scene_to_mesh = False
    args.to_mesh = True
    # load config
    with open(args.config_path, 'r') as stream:
        config = yaml.safe_load(stream)
    voxel_size = config['voxel_size']
    pc_range = config['pc_range']
    occ_size = config['occ_size']

    database_dir = '/media/knight/disk2knight/htmine_occ'
    db = Database(database_dir,
                  sweep=False)
    pc_seman_dir = os.path.join(database_dir, 'labels/pc_seman/')
    pc_bbox_dir = os.path.join(database_dir, 'labels/pc_bbox/')

    # lidar_ego_pose0 = np.load(os.path.join(
    #     pose_path, 'lidar_ego_pose0.npy'), allow_pickle=True).item()
    # lidar_calibrated_sensor0 = np.load(os.path.join(
    #     calib_path, 'lidar_calibrated_sensor0.npy'), allow_pickle=True).item()
    # print(lidar_ego_pose0, lidar_calibrated_sensor0)

    for clip_idx in tqdm(range(0, len(db.clip_stamps.keys()))):
        # clip_idxs = [clip_idx-1, clip_idx, clip_idx+1]
        # 多个 Clip 存在问题，仍然采用只使用当前 clip
        clip_idxs = [clip_idx]
        stamps = []
        for cli_id in clip_idxs:
            if cli_id >= len(db.clip_stamps.keys()) or cli_id < 0:
                continue
            clip_ = list(db.clip_stamps.keys())[cli_id]
            stamps.extend(db.clip_stamps[clip_])
        clip_name = list(db.clip_stamps.keys())[clip_idx]
        if clip_name not in fail_scenes:
            print("Skip scene: ", clip_name)
            continue
        stamps_ = db.clip_stamps[clip_name]
        if len(stamps_) < 10:
            continue
        clip_stamp_min = stamps_[0]
        clip_stamp_max = stamps_[-1]
        print(clip_name, clip_stamp_min, clip_stamp_max)
        dict_list = []
        lidar2odom_0 = db.load_calib(stamps[0])[0].T_lidar_odom
        for stamp in stamps:
            if not osp.exists(os.path.join(pc_seman_dir, f'{stamp}.npy')):
                continue
            pc0 = np.load(os.path.join(pc_seman_dir, f'{stamp}.npy'))[:, :4]
            # cloud_viewer(pc0)
            pc_bbox_path = os.path.join(pc_bbox_dir, f'{stamp}.npy')
            if osp.exists(pc_bbox_path):
                bboxes = np.load(pc_bbox_path)
            else:
                bboxes = np.array([[0, 0, 0., 0., 0., 1., 1., 1., 0.]])
            # print(pc_bbox_path, bboxes)
            boxes = bboxes[:, 2:].astype(np.float32)
            # boxes = boxes
            object_category = bboxes[:, 1].astype(np.uint8)
            boxes_token = bboxes[:, 0]
            locs = boxes[:, 0:3]
            dims = boxes[:, 3:6]
            rots = boxes[:, 6:7]
            boxes[:, 2] -= dims[:, 2] / 2.
            # print(boxes)
            points_in_boxes = points_in_boxes_cpu(torch.from_numpy(pc0[:, :3][np.newaxis, :, :]),
                                                  torch.from_numpy(boxes[np.newaxis, :]))
            # print(torch.sum(points_in_boxes.byte()[:, :, 0], dim=1))
            # print(points_in_boxes.shape)
            object_points_list = []
            j = 0
            while j < points_in_boxes.shape[-1]:
                object_points_mask = points_in_boxes[0][:, j].bool()
                object_points = pc0[object_points_mask]
                object_points_list.append(object_points)
                j = j + 1
            # vis = o3d.visualization.Visualizer()
            # vis.create_window()
            # for obs in boxes:
            #     b = o3d.geometry.OrientedBoundingBox()
            #     b.center = obs[:3]
            #     # b.extent = obs[3:6]
            #     l, w, h, yaw = obs[3:]
            #     b.extent = [l, w, h]
            #     # with heading
            #     R = o3d.geometry.OrientedBoundingBox.get_rotation_matrix_from_xyz(
            #         (0, 0, yaw))
            #     b.rotate(R, b.center)
            #     vis.add_geometry(b)
            moving_mask = torch.ones_like(points_in_boxes)
            points_in_boxes = torch.sum(
                points_in_boxes * moving_mask, dim=-1).bool()
            points_mask = ~(points_in_boxes[0])
            ############################# get point mask of the vehicle itself ##########################
            range = config['self_range']
            oneself_mask = torch.from_numpy((np.abs(pc0[:, 0]) > range[0]) |
                                            (np.abs(pc0[:, 1]) > range[1]) |
                                            (np.abs(pc0[:, 2]) > range[2]))

            ############################# get static scene segment ##########################
            points_mask = points_mask & oneself_mask
            pc = pc0[points_mask]
            # point_cloud = o3d.geometry.PointCloud()
            # point_cloud.points = o3d.utility.Vector3dVector(pc[:, :3])
            # vis.add_geometry(point_cloud)
            # # point_cloud = o3d.geometry.PointCloud()
            # # point_cloud.points = o3d.utility.Vector3dVector(
            # #     object_points[:, :3])
            # # vis.add_geometry(point_cloud)
            # vis.get_render_option().background_color = np.asarray(
            #     [0, 0, 0])  # you can set the bg color
            # vis.run()
            # vis.destroy_window()
            ################## coordinate conversion to the same (first) LiDAR coordinate  ##################
            lidar2odom_t = db.load_calib(stamp)[0].T_lidar_odom
            pc_odom = db.transform_pc(pc, lidar2odom_t)
            lidar_pc = db.transform_pc(pc_odom, np.linalg.inv(lidar2odom_0))
            # cloud_viewer(lidar_pc)
            dict = {"object_tokens": boxes_token,
                    "object_points_list": object_points_list,
                    "lidar_pc": lidar_pc,
                    "calib_lidar2odom": lidar2odom_t,
                    "gt_bbox_3d": boxes,
                    "converted_object_category": object_category,
                    "pc_file_name": stamp}
            dict_list.append(dict)
        ################## concatenate all static scene segments  ########################
        lidar_pc_list = [dict['lidar_pc'] for dict in dict_list]
        lidar_pc = np.concatenate(lidar_pc_list, axis=0)
        # cloud_viewer(lidar_pc)
        ################## concatenate all object segments (including non-key frames)  ########################
        object_token_zoo = []
        object_semantic = []
        for dict in dict_list:
            for i, object_token in enumerate(dict['object_tokens']):
                if object_token not in object_token_zoo:
                    if (dict['object_points_list'][i].shape[0] > 0):
                        object_token_zoo.append(object_token)
                        object_semantic.append(
                            dict['converted_object_category'][i])
                    else:
                        continue
        object_points_dict = {}
        for query_object_token in object_token_zoo:
            object_points_dict[query_object_token] = []
            for dict in dict_list:
                for i, object_token in enumerate(dict['object_tokens']):
                    if query_object_token == object_token:
                        object_points = dict['object_points_list'][i]
                        if object_points.shape[0] > 0:
                            object_points = object_points[:,
                                                          :3] - dict['gt_bbox_3d'][i][:3]
                            rots = dict['gt_bbox_3d'][i][6]
                            Rot = Rotation.from_euler(
                                'z', -rots, degrees=False)
                            rotated_object_points = Rot.apply(object_points)
                            object_points_dict[query_object_token].append(
                                rotated_object_points)
                    else:
                        continue
            object_points_dict[query_object_token] = np.concatenate(object_points_dict[query_object_token],
                                                                    axis=0)
        object_points_vertice = []
        for key in object_points_dict.keys():
            point_cloud = object_points_dict[key]
            object_points_vertice.append(point_cloud[:, :3])
        # cloud_viewer(lidar_pc)
        if args.whole_scene_to_mesh:
            point_cloud_original = o3d.geometry.PointCloud()
            with_normal2 = o3d.geometry.PointCloud()
            point_cloud_original.points = o3d.utility.Vector3dVector(
                lidar_pc[:, :3])
            with_normal = preprocess(point_cloud_original, config)
            with_normal2.points = with_normal.points
            with_normal2.normals = with_normal.normals
            mesh, _ = create_mesh_from_map(None, 11, config['n_threads'],
                                           config['min_density'], with_normal2)
            lidar_pc = np.asarray(mesh.vertices, dtype=float)
            lidar_pc = np.concatenate(
                (lidar_pc, np.ones_like(lidar_pc[:, 0:1])), axis=1)
            # cloud_viewer(lidar_pc)
        i = -1

        pbar = tqdm(total=len(dict_list))
        while int(i) < 10000:  # Assuming the sequence does not have more than 10000 frames
            pbar.update(1)
            i = i + 1
            if i >= len(dict_list):
                print('finish scene!')
                break
            dict = dict_list[i]
            if float(dict['pc_file_name']) < float(clip_stamp_min) or float(dict['pc_file_name']) > float(clip_stamp_max):
                continue

            ################## convert the static scene to the target coordinate system ##############
            # lidar_calibrated_sensor = dict['lidar_calibrated_sensor']
            # lidar_ego_pose = dict['lidar_ego_pose']
            lidar2odom_t = dict['calib_lidar2odom']
            pc_odom = db.transform_pc(lidar_pc, lidar2odom_0)
            lidar_pc_i = db.transform_pc(pc_odom, np.linalg.inv(lidar2odom_t))
            point_cloud = lidar_pc_i[:, :3]
            if args.with_semantic:
                point_cloud_with_semantic = lidar_pc_i[:, :4]

            gt_bbox_3d = dict['gt_bbox_3d']
            locs = gt_bbox_3d[:, 0:3]
            dims = gt_bbox_3d[:, 3:6]
            rots = gt_bbox_3d[:, 6:7]
            # gt_bbox_3d[:, 2] += dims[:, 2] / 2.

            ################## bbox placement ##############
            object_points_list = []
            object_semantic_list = []
            # vis = o3d.visualization.Visualizer()
            # vis.create_window()
            # pc_vis = o3d.geometry.PointCloud()
            # pc_vis.points = o3d.utility.Vector3dVector(
            #     dict['lidar_pc'][:, :3])
            # vis.add_geometry(pc_vis)
            # print(object_token_zoo, dict['object_tokens'])
            for j, object_token in enumerate(dict['object_tokens']):
                for k, object_token_in_zoo in enumerate(object_token_zoo):
                    if object_token == object_token_in_zoo and object_token != '0':
                        points = object_points_vertice[k]
                        Rot = Rotation.from_euler('z', rots[j], degrees=False)
                        rotated_object_points = Rot.apply(points)
                        points = rotated_object_points + locs[j]
                        # pc_vis = o3d.geometry.PointCloud()
                        # pc_vis.points = o3d.utility.Vector3dVector(
                        #     points[:, :3])
                        # vis.add_geometry(pc_vis)
                        if points.shape[0] >= 5:
                            points_in_boxes = points_in_boxes_cpu(torch.from_numpy(points[:, :3][np.newaxis, :, :]),
                                                                  torch.from_numpy(gt_bbox_3d[j:j+1][np.newaxis, :]))
                            points = points[points_in_boxes[0, :, 0].bool()]

                        object_points_list.append(points)
                        semantics = np.ones_like(
                            points[:, 0:1]) * object_semantic[k]
                        object_semantic_list.append(np.concatenate(
                            [points[:, :3], semantics], axis=1))
                        # pc_vis = o3d.geometry.PointCloud()
                        # pc_vis.points = o3d.utility.Vector3dVector(
                        #     points[:, :3])
                        # vis.add_geometry(pc_vis)
                        # for obs in gt_bbox_3d[j:j+1]:
                        #     b = o3d.geometry.OrientedBoundingBox()
                        #     b.center = obs[:3]
                        #     # b.extent = obs[3:6]
                        #     l, w, h, yaw = obs[3:]
                        #     b.extent = [l, w, h]
                        #     # with heading
                        #     R = o3d.geometry.OrientedBoundingBox.get_rotation_matrix_from_xyz(
                        #         (0, 0, yaw))
                        #     b.rotate(R, b.center)
                        #     vis.add_geometry(b)
            # vis.get_render_option().background_color = np.asarray(
            #     [0, 0, 0])  # you can set the bg color
            # vis.run()
            # vis.destroy_window()
            try:  # avoid concatenate an empty array
                temp = np.concatenate(object_points_list)
                scene_points = np.concatenate([point_cloud, temp])
            except:
                scene_points = point_cloud

            if args.with_semantic:
                try:
                    temp = np.concatenate(object_semantic_list)
                    # print('dynamic object cloud: ', temp.shape)
                    # print(temp)
                    scene_semantic_points = np.concatenate(
                        [point_cloud_with_semantic, temp])
                except:
                    scene_semantic_points = point_cloud_with_semantic
            # cloud_viewer(scene_semantic_points)
            point_cloud_range = [0, -38.4, -4, 76.8, 38.4, 5.6]

            ################## remain points with a spatial range ##############
            mask = (scene_points[:, 0] > point_cloud_range[0]) & (scene_points[:, 0] < point_cloud_range[3])\
                & (np.abs(scene_points[:, 1]) < point_cloud_range[4]) \
                & (scene_points[:, 2] > point_cloud_range[2]) & (scene_points[:, 2] < point_cloud_range[5])
            scene_points = scene_points[mask]

            if args.to_mesh and not args.whole_scene_to_mesh:
                try:
                    ################## get mesh via Possion Surface Reconstruction ##############
                    point_cloud_original = o3d.geometry.PointCloud()
                    with_normal2 = o3d.geometry.PointCloud()
                    point_cloud_original.points = o3d.utility.Vector3dVector(
                        scene_points[:, :3])
                    with_normal = preprocess(point_cloud_original, config)
                    with_normal2.points = with_normal.points
                    with_normal2.normals = with_normal.normals
                    mesh, _ = create_mesh_from_map(None, config['depth'], config['n_threads'],
                                                   config['min_density'], with_normal2)
                    scene_points = np.asarray(mesh.vertices, dtype=float)
                except:
                    print(stamp, scene_points.shape)
                    continue
            ################## remain points with a spatial range ##############
            mask = (scene_points[:, 0] > point_cloud_range[0]) & (scene_points[:, 0] < point_cloud_range[3])\
                & (np.abs(scene_points[:, 1]) < point_cloud_range[4]) \
                & (scene_points[:, 2] > point_cloud_range[2]) & (scene_points[:, 2] < point_cloud_range[5])
            scene_points = scene_points[mask]
            # mask = (np.abs(scene_points[:, 0]) < 50.0) & (np.abs(scene_points[:, 1]) < 50.0) \
            #     & (scene_points[:, 2] > -5.0) & (scene_points[:, 2] < 3.0)
            # scene_points = scene_points[mask]

            ################## convert points to voxels ##############
            pcd_np = scene_points
            pcd_np[:, 0] = (pcd_np[:, 0] - pc_range[0]) / voxel_size
            pcd_np[:, 1] = (pcd_np[:, 1] - pc_range[1]) / voxel_size
            pcd_np[:, 2] = (pcd_np[:, 2] - pc_range[2]) / voxel_size
            pcd_np = np.floor(pcd_np).astype(np.int32)

            voxel = np.zeros(occ_size)
            voxel[pcd_np[:, 0], pcd_np[:, 1], pcd_np[:, 2]] = 1

            # ################## convert voxel coordinates to LiDAR system  ##############
            gt_ = voxel
            x = np.linspace(0, gt_.shape[0] - 1, gt_.shape[0])
            y = np.linspace(0, gt_.shape[1] - 1, gt_.shape[1])
            z = np.linspace(0, gt_.shape[2] - 1, gt_.shape[2])
            X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
            vv = np.stack([X, Y, Z], axis=-1)
            fov_voxels = vv[gt_ > 0]
            fov_voxels[:, :3] = (fov_voxels[:, :3] + 0.5) * voxel_size
            fov_voxels[:, 0] += pc_range[0]
            fov_voxels[:, 1] += pc_range[1]
            fov_voxels[:, 2] += pc_range[2]
            # np.save(path + 'occupancy_gt{}.npy'.format(i), fov_voxels)

            if args.with_semantic:

                ################## remain points with a spatial range  ##############
                mask = (scene_semantic_points[:, 0] > point_cloud_range[0]) & (scene_semantic_points[:, 0] < point_cloud_range[3])\
                    & (np.abs(scene_semantic_points[:, 1]) < point_cloud_range[4]) \
                    & (scene_semantic_points[:, 2] > point_cloud_range[2]) & (scene_semantic_points[:, 2] < point_cloud_range[5])
                # mask = (scene_semantic_points[:, 0] > 0.0) & (scene_semantic_points[:, 0] < 100.0)\
                #     & (np.abs(scene_semantic_points[:, 1]) < 40.0) \
                #     & (scene_semantic_points[:, 2] > -4.0) & (scene_semantic_points[:, 2] < 10.0)
                # mask = (np.abs(scene_semantic_points[:, 0]) < 120.0) & (np.abs(scene_semantic_points[:, 1]) < 120.0) \
                #     & (scene_semantic_points[:, 2] > -15.0) & (scene_semantic_points[:, 2] < 15.0)
                scene_semantic_points = scene_semantic_points[mask]

                ################## Nearest Neighbor to assign semantics ##############
                dense_voxels = fov_voxels
                sparse_voxels_semantic = scene_semantic_points

                x = torch.from_numpy(dense_voxels).cuda().unsqueeze(0).float()
                y = torch.from_numpy(
                    sparse_voxels_semantic[:, :3]).cuda().unsqueeze(0).float()
                d1, d2, idx1, idx2 = chamfer.forward(x, y)
                indices = idx1[0].cpu().numpy()

                dense_semantic = sparse_voxels_semantic[:, 3][np.array(
                    indices)]
                dense_voxels_with_semantic = np.concatenate(
                    [fov_voxels, dense_semantic[:, np.newaxis]], axis=1)

                # to voxel coordinate
                pcd_np = dense_voxels_with_semantic
                pcd_np[:, 0] = (pcd_np[:, 0] - pc_range[0]) / voxel_size
                pcd_np[:, 1] = (pcd_np[:, 1] - pc_range[1]) / voxel_size
                pcd_np[:, 2] = (pcd_np[:, 2] - pc_range[2]) / voxel_size
                dense_voxels_with_semantic = np.floor(pcd_np).astype(np.int32)
                # print(dense_voxels_with_semantic)
                stamp = dict['pc_file_name']
                occ_path = osp.join(
                    database_dir, f'labels/pc_occ/{stamp}.npy')
                make_path_dirs(occ_path)
                np.save(occ_path,
                        dense_voxels_with_semantic)
                # visual_occ_htmine(occ_path)

                # cloud_viewer_rgb_id(dense_voxels_with_semantic)
        pbar.close()

        # break
    # print(config)
