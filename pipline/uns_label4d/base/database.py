#!/usr/bin/env python3
# coding=utf-8
"""
brief: 
Version: v0.0.1
Author: knightdby  && knightdby@163.com
Date: 2025-03-25 20:06:43
Description: 
LastEditors: knightdby
LastEditTime: 2025-05-15 18:15:40
FilePath: /UnScenes3D/pipline/uns_label4d/base/database.py
Copyright 2025 by Inc, All Rights Reserved. 
2025-03-25 20:06:43
"""

from manifast import *
from collections import namedtuple
from scipy.spatial.transform import Rotation as R
from pipline.uns_label4d.base.calib import Calibration
import open3d as o3d
from shapely.geometry import Point, Polygon
# import mayavi.mlab
from shapely.prepared import prep
# from tvtk.api import tvtk
classes_2d_seg = ['background', 'truck', 'widebody', 'car',
                  'excavator', 'machinery', 'person',
                  'warningsign', 'signboard', 'fence',
                  'cable', 'rock', 'puddle',
                  'slit', 'rut', 'building',
                  'retainingwall', 'road']

class_to_name_seg = {index: value for index,
                     value in enumerate(classes_2d_seg)}
name_to_class_seg = {v: n for n, v in class_to_name_seg.items()}
# RGB(255, 255, 255), RGB(0, 100, 128), RGB(0, 150, 128), RGB(0, 200, 128)
# RGB(0, 255, 128), RGB(0, 255, 255), RGB(128, 0, 0),
# RGB(128, 0, 64), RGB(128, 0, 128), RGB(64, 0, 128),
# RGB(128, 0, 255), RGB(255, 128, 0), RGB(0, 0, 255),
# RGB(64, 64, 0), RGB(255,  255, 0), RGB(128, 64, 128),
# RGB(128, 128, 128), RGB(0, 128, 200),
palette = [(255, 255, 255), (0, 100, 128), (0, 150, 128), (0, 200, 128),
           (0, 255, 128), (0, 255, 255), (128, 0, 0),
           (128, 0, 64), (128, 0, 128), (64, 0, 128),
           (128, 0, 255), (255, 128, 0), (0, 0, 255),
           (200, 200, 0), (255,  255, 0), (128, 64, 128),
           (128, 128, 128), (0, 128, 200),
           ]


class Database:
    def __init__(self, data_dir, sweep=False):
        self.data_dir = data_dir
        self.obs_type = set()
        self.invalid_scenes = ['scene_00181', 'scene_00182', 'scene_00185', 'scene_00186',
                               'scene_00187', 'scene_00188', 'scene_00189', 'scene_00190',
                               'scene_00191', 'scene_00192', 'scene_00351', 'scene_00356',
                               'scene_00357', 'scene_00413', 'scene_00436',
                               'scene_00462'
                               ]
        self._load_timestamps(sweep)
        self.label_bevmaps = {}
        bevmap_label_path = osp.join(
            self.data_dir, 'labels/label_bevmap', 'label_bevmap.json')
        if osp.exists(bevmap_label_path):
            label_bevmaps = read_json_data(bevmap_label_path)
            for maps in label_bevmaps:
                name = maps['pcd'][:-4]
                self.label_bevmaps[name] = maps['labels']
        bevmap_label_path = osp.join(
            self.data_dir, 'labels/label_bevmap', 'label_bevmap_2.json')
        if osp.exists(bevmap_label_path):
            label_bevmaps = read_json_data(bevmap_label_path)
            for maps in label_bevmaps:
                name = maps['pcd'][:-4]
                self.label_bevmaps[name] = maps['labels']
        # self.label_bevmaps['2'] = self.label_bevmaps['scene_00480']
        # self.label_bevmaps['2'].extend(self.label_bevmaps['scene_00481'])

    def _load_timestamps(self, sweep):
        clip_info_path = osp.join(
            self.data_dir, 'infos/clips_info.json')
        with open(clip_info_path, "r", encoding='utf-8') as f:
            json_data = json.load(f)
        self.clip_stamps = {}
        for clip, value in json_data.items():
            frames = []
            frames.extend(value['samples'])
            if sweep:
                frames.extend(value['sweeps'])
            frames = sorted(list(set(frames)))
            if clip in self.invalid_scenes:
                frames = []
            self.clip_stamps[clip] = frames

    def load_calib(self, timestamp):
        data = {}
        filepath = os.path.join(
            self.data_dir, f'samples/calibs/{timestamp}.txt')
        # make_path_dirs('./calibs_00071')
        # save_path = os.path.join('./calibs_00071', f'{timestamp}.txt')
        # shutil.copy(filepath, save_path)
        if not osp.exists(filepath):
            filepath = filepath.replace('samples', 'sweeps')
        calib_data = {}
        with open(filepath, 'r') as f:
            for line in f.readlines():
                if ':' in line:
                    key, value = line.split(':', 1)
                    calib_data[key] = np.array(
                        [float(x) for x in value.split()])
        P2_param = calib_data['P2'].reshape(3, 4)
        R0_rect = calib_data['R0_rect'].reshape(3, 3)
        # 加载 Tr_velo_to_cam (3x4 矩阵)
        Tr_velo_to_cam = calib_data['Tr_velo_to_cam'].reshape(3, 4)
        # 将 Tr_velo_to_cam 转换为 4x4 矩阵
        Tr_velo_to_cam_hom = np.eye(4)
        Tr_velo_to_cam_hom[:3, :] = Tr_velo_to_cam
        data['T_lidar_camera'] = Tr_velo_to_cam_hom
        Tr_velo_to_imu = calib_data['Tr_velo_to_imu'].reshape(3, 4)
        Tr_velo_to_imu_hom = np.eye(4)
        Tr_velo_to_imu_hom[:3, :] = Tr_velo_to_imu
        data['T_lidar_odom'] = Tr_velo_to_imu_hom
        data['P2'] = P2_param
        # 使用 gps odom 进行全局地图构建
        # data['T_lidar_odom'] = self.load_egopose_tf(timestamp)
        # data['T_lidar_odom'] = self.load_odompose_tf(timestamp)

        calib = Calibration(filepath)

        return namedtuple('CalibData', data.keys())(*data.values()), calib

    def load_egopose_tf(self, stamp):
        location_path = os.path.join(
            self.data_dir, f'samples/ego_pose/{stamp}.txt')
        if not osp.exists(location_path):
            location_path = location_path.replace('samples', 'sweeps')

        lines = []
        with open(location_path, 'r') as file_to_read:
            while True:
                line = file_to_read.readline()  # 整行读取数据
                lines.append(line)
                if not line:
                    break
        key, value = lines[0].split(' ', 1)
        rt_param = np.array([float(x) for x in value.split()])
        quaternion = rt_param[3:]
        translation = rt_param[:3]
        r = R.from_quat(quaternion)
        rotation_matrix = r.as_matrix()
        trans = np.eye(4)
        trans[:3, :3] = rotation_matrix
        trans[:3, 3] = translation
        return trans

    def load_odompose_tf(self, stamp):
        location_path = os.path.join(
            self.data_dir, f'odom_pose/{stamp}.txt')

        lines = []
        with open(location_path, 'r') as file_to_read:
            while True:
                line = file_to_read.readline()  # 整行读取数据
                lines.append(line)
                if not line:
                    break
        key, value = lines[0].split(' ', 1)
        rt_param = np.array([float(x) for x in value.split()])
        quaternion = rt_param[3:]
        translation = rt_param[:3]
        r = R.from_quat(quaternion)
        rotation_matrix = r.as_matrix()
        trans = np.eye(4)
        trans[:3, :3] = rotation_matrix
        trans[:3, 3] = translation
        return trans

    def load_lidar(self, lidar_stamp, view=False):
        lidar_path = osp.join(
            self.data_dir, 'samples/clouds', lidar_stamp+'.bin')
        if not osp.exists(lidar_path):
            lidar_path = lidar_path.replace('samples', 'sweeps')
        pointcloud = np.fromfile(lidar_path, dtype=np.float32)
        pointcloud = pointcloud.reshape(-1, 4)
        return pointcloud

    def load_camera(self, stamp):
        img_path = os.path.join(
            self.data_dir, f'samples/images/{stamp}.jpg')
        if not osp.exists(img_path):
            img_path = img_path.replace('samples', 'sweeps')
        image = cv2.imread(img_path)
        return image

    def load_label2d(self, stamp):
        label_path = os.path.join(
            self.data_dir, f'labels/label_2d/{stamp}.json')
        if osp.exists(label_path):
            obs_array = read_json_data(label_path)
            return obs_array
        else:
            return None

    def load_labelsegmap(self, stamp):
        label_path = os.path.join(
            self.data_dir, f'labels/label_segmap/{stamp}.png')
        if osp.exists(label_path):
            segmap = cv2.imread(label_path, 0)
            segmap_rgb = np.zeros(
                (segmap.shape[0], segmap.shape[1], 3), dtype=np.uint8)
            for label_idx in range(len(palette)):
                # if label_idx in [1, 2, 3, 4, 5, 6, 11]:
                #     segmap_rgb[segmap == label_idx] = (0, 0, 0)
                # else:
                #     segmap_rgb[segmap == label_idx] = palette[label_idx]
                segmap_rgb[segmap == label_idx] = palette[label_idx]
            return segmap_rgb
        else:
            return None

    def load_pc_seman(self, stamp):
        label_path = os.path.join(
            self.data_dir, f'labels/pc_seman/{stamp}.npy')
        if osp.exists(label_path):
            pc_seman = np.load(label_path)
            return pc_seman
        else:
            return None

    def load_label3d(self, stamp):
        label_path = os.path.join(
            self.data_dir, f'labels/label_3d_lidar/{stamp}.txt')
        obs_array = []
        if osp.exists(label_path):
            obs_array = read_txt_data(label_path)
        obs_out = []
        for obs in obs_array:
            obs_info = obs.split(' ')
            label = obs_info[0]
            pad_value = 0.6
            # pad_value = 0.
            h, w, l = float(obs_info[8])+pad_value, float(
                obs_info[9])+pad_value, float(obs_info[10])+pad_value
            x, y, z = float(obs_info[11]), float(
                obs_info[12]), float(obs_info[13])
            if label == 'truck' and w < 5.:
                label = 'widebody'
            elif label == 'widebody' and w > 6.:
                label = 'truck'
                # print(label_path.replace('labels/label_3d_lidar',
                #       'samples/images').replace('.txt', '.jpg'))
            orient = float(obs_info[14])
            token = obs_info[-1]
            obs_i = [token, label, x, y, z, w, h, l, orient]
            obs_out.append(obs_i)
        return obs_out

    def load_label4d(self, clip):
        label_path = os.path.join(
            self.data_dir, f'labels/label_4d/{clip}.txt')
        obs_array = []
        if not osp.exists(label_path):
            label_path = label_path.replace('scene_0', 'clip_')
        if not osp.exists(label_path):
            return []
        obs_array = read_txt_data(label_path)
        obs_out = []
        for obs in obs_array:
            obs_info = obs.split(' ')
            label = obs_info[0]
            # 伊敏特殊情况，建筑物标注为了栅栏
            if int(clip.split('_')[1]) > 294 and int(clip.split('_')[1]) < 381:
                if label == 'fence':
                    label = 'building'
            if 'warning' in obs:
                obs_info = obs_info[1:]
            pad_value = 0.6
            if label == 'building':
                pad_value = 0.
            # pad_value = 0.
            h, w, l = float(obs_info[8])+pad_value, float(
                obs_info[9])+pad_value, float(obs_info[10])+pad_value
            x, y, z = float(obs_info[11]), float(
                obs_info[12]), float(obs_info[13])
            # orient = float(obs_info[3])
            orient = float(obs_info[14])
            token = obs_info[-1]
            obs_i = [token, label, x, y, z, w, h, l, orient]
            obs_out.append(obs_i)
        return obs_out

    def load_label4d_invis(self, clip):
        label_path = os.path.join(
            self.data_dir, f'labels/label_4d_invis/{clip}.txt')
        obs_array = []
        if osp.exists(label_path):
            obs_array = read_txt_data(label_path)
        obs_out = []
        for obs in obs_array:
            obs_info = obs.split(' ')
            label = obs_info[0]
            # if 'warning' in obs:
            #     obs_info = obs_info[1:]
            # pad_value = 0.6
            pad_value = 0.
            h, w, l = float(obs_info[8])+pad_value, float(
                obs_info[9])+pad_value, float(obs_info[10])+pad_value
            x, y, z = float(obs_info[11]), float(
                obs_info[12]), float(obs_info[13])
            orient = float(obs_info[14])
            token = obs_info[-1]
            obs_i = [token, label, x, y, z, w, h, l, orient]
            # warningsign 会存在投影偏差
            if 'rock' in label:
                obs_out.append(obs_i)
        return obs_out

    def load_bevmap(self, clip):
        if clip in self.label_bevmaps:
            bevmaps = self.label_bevmaps[clip]
            outputs_ord = []
            for obj in bevmaps:
                if obj['label'] == '路面':
                    outputs_ord.append(obj)
            for obj in bevmaps:
                if obj['label'] != '路面':
                    outputs_ord.append(obj)
            return outputs_ord
        else:
            return []

    def split_dyna_static_cloud(self, pointcloud, dynamic_obss, view=False):
        if len(dynamic_obss) == 0:
            view = False
        if view:
            vis = o3d.visualization.Visualizer()
            vis.create_window()
        obs_cloud = np.array([[0, 0, 0, 0, 0, 0]])
        for obs in dynamic_obss:
            # 矩形框的参数：中心点 (x, y, z) 和尺寸 (w, h, l)
            bbox_center = np.array(obs[2:5])  # 矩形框的中心 (x, y, z)
            w, h, l, yaw = obs[5:]  # 矩形框的宽度、高度和长度 yaw
            w += 2
            h += 2
            # yaw = -yaw
            # yaw = np.radians(45)  # 45 度绕 z 轴旋转
            # 1. 提取点云的 xyz 部分
            xyz_points = pointcloud[:, :3]
            # 2. 将点云平移到以 BBox 中心为原点的坐标系
            translated_points = xyz_points - bbox_center
            # 3. 计算绕 z 轴的旋转矩阵（二维旋转，只考虑平面上的 yaw 角）
            rotation_matrix = np.array([
                [np.cos(-yaw), -np.sin(-yaw), 0],
                [np.sin(-yaw),  np.cos(-yaw), 0],
                [0, 0, 1]
            ])
            # 4. 将点云旋转到 BBox 的本地坐标系
            rotated_points = np.dot(translated_points, rotation_matrix.T)
            # 5. 在本地坐标系中计算 BBox 的边界
            # x_min, x_max = -w / 2, w / 2
            # y_min, y_max = -h / 2, h / 2
            # z_min, z_max = -l / 2, l / 2
            x_min, x_max = -l / 2, l / 2
            y_min, y_max = -w / 2, w / 2
            z_min, z_max = -h / 2, h / 2+5
            # 6. 筛选位于 BBox 内的点 (只考虑 xyz 部分)
            inside_mask = (
                (rotated_points[:, 0] >= x_min) & (rotated_points[:, 0] <= x_max) &
                (rotated_points[:, 1] >= y_min) & (rotated_points[:, 1] <= y_max) &
                (rotated_points[:, 2] >= z_min) & (
                    rotated_points[:, 2] <= z_max)
            )
            points_in_box = np.array(pointcloud[inside_mask])
            points_in_box = np.hstack(
                (points_in_box[:, :3],
                 np.zeros((points_in_box.shape[0], 1)),
                 np.ones((points_in_box.shape[0], 1))*100,
                 np.ones((points_in_box.shape[0], 1))*128))
            obs_cloud = np.concatenate(
                (obs_cloud, points_in_box), axis=0)
            pointcloud = pointcloud[~inside_mask]
            # print(points_in_box.shape)
            pointcloud = np.array(pointcloud)
            # cloud_viewer(pointcloud)

            # 可视化 bbox
            if view:
                # draw_boxes(vis, boxes, (0, 1, 0))
                b = o3d.geometry.OrientedBoundingBox()
                b.center = obs[2:5]
                w, h, l, yaw = obs[5:]
                b.extent = [l, w, h]
                # with heading
                R = o3d.geometry.OrientedBoundingBox.get_rotation_matrix_from_xyz(
                    (0, 0, yaw))
                b.rotate(R, b.center)
                vis.add_geometry(b)
        if view:
            point_cloud = o3d.geometry.PointCloud()
            point_cloud.points = o3d.utility.Vector3dVector(pointcloud[:, :3])
            vis.add_geometry(point_cloud)
            point_cloud = o3d.geometry.PointCloud()
            point_cloud.points = o3d.utility.Vector3dVector(obs_cloud[:, :3])
            vis.add_geometry(point_cloud)
            vis.get_render_option().background_color = np.asarray(
                [0, 0, 0])  # you can set the bg color
            vis.run()
            vis.destroy_window()
        return obs_cloud[1:], pointcloud

    def split_static_bbox_cloud(self, cloud_map, static_obs, prepared_polygons=None):
        obs_cloud = []
        if prepared_polygons is not None:
            return np.array(obs_cloud), cloud_map
        for obs in static_obs:
            # 矩形框的参数：中心点 (x, y, z) 和尺寸 (w, h, l)
            bbox_center = np.array(obs[2:5])  # 矩形框的中心 (x, y, z)
            if prepared_polygons is not None:
                p = Point(bbox_center[0], bbox_center[1])
                for prepared_polygon, color in prepared_polygons:
                    # print(bbox_center, prepared_polygon)
                    if prepared_polygon.contains(p) and color == (128, 128, 128):
                        # print('in wall')
                        continue
                continue

            w, h, l, yaw = obs[5:]  # 矩形框的宽度、高度和长度 yaw
            # print(obs)
            if 'building' in obs[1]:
                color = palette[15]
            elif 'warning' in obs[1]:
                color = palette[7]
            elif 'signboard' in obs[1]:
                color = palette[8]
            elif 'fence' in obs[1]:
                color = palette[9]
            elif 'rock' in obs[1]:
                color = palette[11]
            # if 'signboard' in obs['obj_type']:
            #     w, h, l = 3, 6, 3
            # elif 'warningsign' in obs['obj_type']:
            #     w,  l = 1, 1
            # yaw = np.radians(45)  # 45 度绕 z 轴旋转
            # 1. 提取点云的 xyz 部分
            xyz_points = cloud_map[:, :3]
            # 2. 将点云平移到以 BBox 中心为原点的坐标系
            translated_points = xyz_points - bbox_center
            # 3. 计算绕 z 轴的旋转矩阵（二维旋转，只考虑平面上的 yaw 角）
            rotation_matrix = np.array([
                [np.cos(-yaw), -np.sin(-yaw), 0],
                [np.sin(-yaw),  np.cos(-yaw), 0],
                [0, 0, 1]
            ])
            # 4. 将点云旋转到 BBox 的本地坐标系
            rotated_points = np.dot(translated_points, rotation_matrix.T)
            # 5. 在本地坐标系中计算 BBox 的边界
            # x_min, x_max = -w / 2, w / 2
            # y_min, y_max = -h / 2, h / 2
            # z_min, z_max = -l / 2, l / 2
            x_min, x_max = -l / 2, l / 2
            y_min, y_max = -w / 2, w / 2
            z_min, z_max = -h / 2, h / 2+5.  # 扩充一下高度，使得在高度上也能筛选到点
            # 6. 筛选位于 BBox 内的点 (只考虑 xyz 部分)
            inside_mask = (
                (rotated_points[:, 0] >= x_min) & (rotated_points[:, 0] <= x_max) &
                (rotated_points[:, 1] >= y_min) & (rotated_points[:, 1] <= y_max) &
                (rotated_points[:, 2] >= z_min) & (
                    rotated_points[:, 2] <= z_max)
            )

            points_in_box = cloud_map[inside_mask]
            cloud_map = cloud_map[~inside_mask]

            cloud_map = np.array(cloud_map)
            for point in points_in_box:
                point[3:] = color
                obs_cloud.append(point)
        non_obs_cloud = cloud_map
        return np.array(obs_cloud), non_obs_cloud

    def prepare_bevmap_polygons(self, bevmaps):
        # 优先级顺序：挡墙 > 车辙 > 水坑 > 路面
        priority = ['挡墙', '车辙', '水坑', '路面']
        label_color_map = {
            '挡墙': (128, 128, 128),
            '车辙': (255, 255, 0),
            '水坑': (0, 0, 255),
            '路面': (0, 128, 200),
        }

        # 提前构建并准备 polygon（按优先级）
        prepared_polygons = []
        for label in priority:
            for bevmap in bevmaps:
                if bevmap['label'] != label:
                    continue
                # polygon_points = np.array(bevmap['points']).reshape(-1, 2)
                polygon_points = np.array(bevmap['points'])[:, :2]
                polygon = Polygon(polygon_points)
                if polygon.area >= 0.5:
                    prepared_polygons.append((
                        prep(polygon),   # prepared polygon for faster contains
                        label_color_map[label]
                    ))
        return prepared_polygons

    def split_static_bevmap_cloud(self, cloud_map, prepared_polygons):

        obs_cloud = []
        non_obs_cloud = []

        for point in cloud_map:
            p = Point(point[0], point[1])
            matched = False
            for prepared_polygon, color in prepared_polygons:
                if prepared_polygon.contains(p):
                    point[3:] = color
                    obs_cloud.append(point)
                    matched = True
                    break
            if not matched:
                point[3:] = (255, 255, 255)
                non_obs_cloud.append(point)

        return np.array(obs_cloud), np.array(non_obs_cloud)

    def proj_point2image(self, pointcloud, Tr_velo_to_cam, P2, image, segmap_rgb, enable_outline=False):
        pointcloud = pointcloud[pointcloud[:, 0] > 1]
        pointcloud = pointcloud[pointcloud[:, 0] < 120]
        velo_points_hom = np.hstack(
            (pointcloud[:, :3], np.ones((pointcloud.shape[0], 1))))
        cam_points = Tr_velo_to_cam.dot(velo_points_hom.T).T
        img_points_hom = P2.dot(cam_points.T).T
        img_points = img_points_hom[:, :2] / img_points_hom[:, 2, np.newaxis]
        img_h, img_w, _ = image.shape
        image_pt = image.copy()
        in_image_bounds = (
            (img_points[:, 0] >= 0) & (img_points[:, 0] < img_w) &
            (img_points[:, 1] >= img_h*0) & (img_points[:, 1] < img_h)
        )
        if enable_outline:
            img_points_in = img_points
            pointcloud_in = pointcloud
        else:
            img_points_in = img_points[in_image_bounds]
            pointcloud_in = pointcloud[in_image_bounds]
        colors = np.zeros((img_points_in.shape[0], 3), dtype=np.uint8)
        colors_seg = np.zeros((img_points_in.shape[0], 3), dtype=np.uint8)
        pointcloud_valid = np.ones((pointcloud_in.shape[0]), dtype=np.bool8)
        for i, point in enumerate(img_points_in):
            u, v = int(point[0]), int(point[1])
            if 0 <= u < img_w and 0 <= v < img_h:
                colors[i] = image[v, u, ::-1]
                colors_seg[i] = segmap_rgb[v, u, :]
                cv2.circle(image_pt, (u, v), 2, (0, 255, 0), -1)
                r, g, b = segmap_rgb[v, u, :]
                if (r, g, b) == (0, 0, 0):
                    pointcloud_valid[i] = False
                # elif (r, g, b) == (255, 128, 0):
                    # pointcloud_valid[i] = False

            elif not enable_outline:
                pointcloud_valid[i] = False
            else:
                # RGB(200,200,200)
                colors_seg[i] = (200, 200, 200)
        pointcloud_rgb = np.hstack((pointcloud_in, colors))[pointcloud_valid]
        pointcloud_seg_rgb = np.hstack((pointcloud_in, colors_seg))[
            pointcloud_valid]
        return image_pt, pointcloud_rgb, pointcloud_seg_rgb

    def build_static_cloud(self, stamp, pad_cloud=[]):
        calib = self.load_calib(stamp)[0]
        dynamic_obss = self.load_label3d(stamp)
        lidar_pc = self.load_lidar(stamp)
        lidar_pc = lidar_pc[lidar_pc[:, 0] < 110, :]
        lidar_pc = np.hstack(
            (lidar_pc[:, :3], np.ones((lidar_pc.shape[0], 3))))
        _, points = self.split_dyna_static_cloud(
            lidar_pc, dynamic_obss)
        points = points[:, :3]

        if len(pad_cloud) > 0:
            points = np.concatenate(
                (points, pad_cloud), axis=0)
        image_path = osp.join(
            self.data_dir, 'samples/images', f'{stamp}.jpg')
        segmap_path = osp.join(
            self.data_dir, 'labels/label_segmap', f'{stamp}.png')
        if not osp.exists(image_path):
            image_path = image_path.replace('samples', 'sweeps')
            segmap_path = segmap_path.replace('samples', 'sweeps')
        image = cv2.imread(image_path)
        segmap = cv2.imread(segmap_path, 0)
        segmap_rgb = np.zeros_like(image, dtype=np.uint8)
        segmap_rgb_ = np.zeros_like(image, dtype=np.uint8)
        for label_idx in range(len(palette)):
            if label_idx in [1, 2, 3, 4, 5, 6, 11]:
                segmap_rgb_[segmap == label_idx] = (0, 0, 0)
            else:
                segmap_rgb_[segmap == label_idx] = palette[label_idx]
            segmap_rgb[segmap == label_idx] = palette[label_idx]
        projected_image, pointcloud_rgb, pointcloud_seg_rgb = self.proj_point2image(
            points, calib.T_lidar_camera, calib.P2, image, segmap_rgb_, enable_outline=False)

        return image, segmap_rgb, pointcloud_rgb, pointcloud_seg_rgb

    def build_static_cloudmap4clip(self, clip, stamps=[]):
        if len(stamps) < 1:
            stamps = self.clip_stamps[clip]
        map_points = None
        for stamp in stamps:
            image, segmap_rgb, pointcloud_rgb, pointcloud_seg_rgb = self.build_static_cloud(
                stamp)
            map_pts = pointcloud_seg_rgb
            map_pts = self.transform_pc(
                map_pts, self.load_calib(stamp)[0].T_lidar_odom)
            if map_points is None:
                map_points = map_pts
            else:
                map_points = np.concatenate(
                    (map_points, map_pts), axis=0)
            map_points = map_points.astype(np.float32)
        return map_points

    def transform_pc(self, pointcloud, Tr_velo_to_imu):
        points_in_velo = np.hstack(
            (pointcloud[:, :3], np.ones((pointcloud.shape[0], 1))))
        points_in_velo_hom = np.hstack(
            (points_in_velo[:, :3], np.ones((points_in_velo.shape[0], 1))))
        points_in_imu = Tr_velo_to_imu.dot(points_in_velo_hom.T).T
        pointcloud_imu = np.hstack(
            (points_in_imu[:, :3], pointcloud[:, 3:]))
        return pointcloud_imu


def cloud_viewer(pointcloud, path=None, figure='pc_view'):
    import mayavi.mlab
    x = pointcloud[:, 0]  # x position of point
    y = pointcloud[:, 1]  # y position of point
    z = pointcloud[:, 2]  # z position of point
    r = pointcloud[:, 3]  # reflectance value of point
    d = np.sqrt(x ** 2 + y ** 2)  # Map Distance from sensor
    degr = np.degrees(np.arctan(z / d))
    vals = 'height'
    if vals == "height":
        col = z
    else:
        col = d

    # col = r
    fig = mayavi.mlab.figure(
        figure=figure, bgcolor=(0, 0, 0), size=(640*2, 400*2))
    mayavi.mlab.points3d(x, y, z,
                         col,  # Values used for Color
                         mode="point",
                         colormap='spectral',  # 'bone', 'copper', 'gnuplot'
                         # color=(0, 1, 0),   # Used a fixed (r,g,b) instead
                         figure=fig,
                         )
    fig_2 = mayavi.mlab.figure(
        figure=f'{figure}_EvelCloud', bgcolor=(0, 0, 0), size=(640, 400*2))
    mayavi.mlab.points3d(x, y, z,
                         col,  # Values used for Color
                         mode="point",
                         colormap='spectral',  # 'bone', 'copper', 'gnuplot'
                         # color=(0, 1, 0),   # Used a fixed (r,g,b) instead
                         figure=fig_2,
                         )
    if path:
        # scene = fig.scene
        # scene.camera.focal_point = [
        #     -20.4, -0.0,  34.01378558]
        # scene.camera.position = [-22.75131739, -0.0,  35.71378558]
        # # scene.camera.position = [-30.75131739,  -0.78265103, 16.21378558]
        # # scene.camera.focal_point = [-15.25131739,  -0.78265103, 12.21378558]
        # scene.camera.view_angle = 40.0
        # scene.camera.view_up = [1.0, 0.0, 0.0]
        # scene.camera.clipping_range = [0.01, 300.]
        # scene.camera.compute_view_plane_normal()
        # scene.render()
        mayavi.mlab.savefig(path)
        mayavi.mlab.close()
    else:
        mayavi.mlab.show()


def cloud_viewer_save(pointcloud, path=None,  figure='pc_view'):
    import mayavi.mlab
    x = pointcloud[:, 0]  # x position of point
    y = pointcloud[:, 1]  # y position of point
    z = pointcloud[:, 2]  # z position of point
    r = pointcloud[:, 3]  # reflectance value of point
    d = np.sqrt(x ** 2 + y ** 2)  # Map Distance from sensor
    degr = np.degrees(np.arctan(z / d))
    vals = 'height'
    if vals == "height":
        col = z
    else:
        col = d

    # col = r
    fig = mayavi.mlab.figure(
        figure=figure, bgcolor=(0, 0, 0), size=(640*2, 400*2))
    mayavi.mlab.points3d(x, y, z,
                         col,  # Values used for Color
                         mode="point",
                         colormap='spectral',  # 'bone', 'copper', 'gnuplot'
                         # color=(0, 1, 0),   # Used a fixed (r,g,b) instead
                         figure=fig,
                         )
    if path:
        # scene = fig.scene
        # scene.camera.focal_point = [
        #     -20.4, -0.0,  34.01378558]
        # scene.camera.position = [-22.75131739, -0.0,  35.71378558]
        # # scene.camera.position = [-30.75131739,  -0.78265103, 16.21378558]
        # # scene.camera.focal_point = [-15.25131739,  -0.78265103, 12.21378558]
        # scene.camera.view_angle = 40.0
        # scene.camera.view_up = [1.0, 0.0, 0.0]
        # scene.camera.clipping_range = [0.01, 300.]
        # scene.camera.compute_view_plane_normal()
        # scene.render()
        make_path_dirs(path)
        mayavi.mlab.savefig(path)
        mayavi.mlab.close()
    else:
        mayavi.mlab.show()


def cloud_viewer_rgb(pointcloud, pointcloud_2=None, figure='view'):
    import mayavi.mlab
    from tvtk.api import tvtk
    x = pointcloud[:, 0]  # x position of point
    y = pointcloud[:, 1]  # y position of point
    z = pointcloud[:, 2]  # z position of point
    # r = pointcloud[:, 3]  # reflectance value of point
    d = np.sqrt(x ** 2 + y ** 2)  # Map Distance from sensor
    degr = np.degrees(np.arctan(z / d))
    vals = 'height'
    if vals == "height":
        col = z
    else:
        col = d

    colors = pointcloud[:, 3:6]
    colors_with_alpha = np.hstack((colors, 255*np.ones((colors.shape[0], 1))))
    sc = tvtk.UnsignedCharArray()
    sc.from_array(colors_with_alpha)

    fig = mayavi.mlab.figure(
        figure=f'{figure}_SemiCloud', bgcolor=(0, 0, 0), size=(640, 400*2))
    G = mayavi.mlab.points3d(x, y, z, mode='point',
                             scale_factor=1, figure=fig)
    G.mlab_source.dataset.point_data.scalars = sc
    G.mlab_source.dataset.modified()

    if not pointcloud_2 is None:
        x = pointcloud_2[:, 0]  # x position of point
        y = pointcloud_2[:, 1]  # y position of point
        z = pointcloud_2[:, 2]  # z position of point
        colors = pointcloud_2[:, 3:6]
        colors_with_alpha = np.hstack(
            (colors, 255*np.ones((colors.shape[0], 1))))
        sc = tvtk.UnsignedCharArray()
        sc.from_array(colors_with_alpha)
        fig_3 = mayavi.mlab.figure(
            figure=f'{figure}_SemiCloud_2', bgcolor=(0, 0, 0), size=(640, 400*2))
        G2 = mayavi.mlab.points3d(x, y, z, mode='point',
                                  scale_factor=1, figure=fig_3)
        G2.mlab_source.dataset.point_data.scalars = sc
        G2.mlab_source.dataset.modified()
    mayavi.mlab.show()


def cloud_rgb_viewer_save(pointcloud, path=None, figure='pc_view'):
    import mayavi.mlab
    from tvtk.api import tvtk
    x = pointcloud[:, 0]  # x position of point
    y = pointcloud[:, 1]  # y position of point
    z = pointcloud[:, 2]  # z position of point
    r = pointcloud[:, 3]  # reflectance value of point
    d = np.sqrt(x ** 2 + y ** 2)  # Map Distance from sensor
    degr = np.degrees(np.arctan(z / d))
    vals = 'height'
    if vals == "height":
        col = z
    else:
        col = d

    # col = r
    colors = pointcloud[:, 3:6]
    colors_with_alpha = np.hstack((colors, 255*np.ones((colors.shape[0], 1))))
    sc = tvtk.UnsignedCharArray()
    sc.from_array(colors_with_alpha)

    fig = mayavi.mlab.figure(
        figure=f'{figure}_SemiCloud', bgcolor=(0, 0, 0), size=(640, 400*2))
    G = mayavi.mlab.points3d(x, y, z, mode='point',
                             scale_factor=1, figure=fig)
    G.mlab_source.dataset.point_data.scalars = sc
    G.mlab_source.dataset.modified()

    if path:
        # scene = fig.scene
        # scene.camera.focal_point = [
        #     -20.4, -0.0,  34.01378558]
        # scene.camera.position = [-22.75131739, -0.0,  35.71378558]
        # # scene.camera.position = [-30.75131739,  -0.78265103, 16.21378558]
        # # scene.camera.focal_point = [-15.25131739,  -0.78265103, 12.21378558]
        # scene.camera.view_angle = 40.0
        # scene.camera.view_up = [1.0, 0.0, 0.0]
        # scene.camera.clipping_range = [0.01, 300.]
        # scene.camera.compute_view_plane_normal()
        # scene.render()
        make_path_dirs(path)
        mayavi.mlab.savefig(path)
        mayavi.mlab.close()
    else:
        mayavi.mlab.show()


def cloud_viewer_rgb_id(pointcloud, pointcloud_2=None):
    import mayavi.mlab
    from tvtk.api import tvtk
    x = pointcloud[:, 0]  # x position of point
    y = pointcloud[:, 1]  # y position of point
    z = pointcloud[:, 2]  # z position of point
    # r = pointcloud[:, 3]  # reflectance value of point
    d = np.sqrt(x ** 2 + y ** 2)  # Map Distance from sensor
    degr = np.degrees(np.arctan(z / d))
    vals = 'height'
    if vals == "height":
        col = z
    else:
        col = d

    fig_2 = mayavi.mlab.figure(
        figure='EvelCloud', bgcolor=(0, 0, 0), size=(640, 400*2))
    mayavi.mlab.points3d(x, y, z,
                         col,  # Values used for Color
                         mode="point",
                         colormap='spectral',  # 'bone', 'copper', 'gnuplot'
                         # color=(0, 1, 0),   # Used a fixed (r,g,b) instead
                         figure=fig_2,
                         )

    x = pointcloud[:, 0]  # x position of point
    y = pointcloud[:, 1]  # y position of point
    z = pointcloud[:, 2]  # z position of point
    # 确保pointcloud[:, 3]中的每个元素都是有效的索引
    valid_indices = [i for i in pointcloud[:, 3].astype(
        np.uint8) if i < len(palette)]
    # 使用有效索引从palette中获取颜色
    colors = np.array([palette[i] for i in valid_indices])
    # colors = palette[pointcloud[:, 3]]
    colors_with_alpha = np.hstack(
        (colors, 255*np.ones((colors.shape[0], 1))))
    sc = tvtk.UnsignedCharArray()
    sc.from_array(colors_with_alpha)
    fig_3 = mayavi.mlab.figure(
        figure='SemiCloud_2', bgcolor=(0, 0, 0), size=(640, 400*2))
    G2 = mayavi.mlab.points3d(x, y, z, mode='point',
                              scale_factor=1, figure=fig_3)
    G2.mlab_source.dataset.point_data.scalars = sc
    G2.mlab_source.dataset.modified()

    mayavi.mlab.show()
