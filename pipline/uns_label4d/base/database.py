#!/usr/bin/env python3
# coding=utf-8
"""
brief: 
Version: v0.0.1
Author: knightdby  && knightdby@163.com
Date: 2025-03-25 20:06:43
Description: 
LastEditors: knightdby
LastEditTime: 2025-05-19 17:00:12
FilePath: /UnScenes3D/pipline/uns_label4d/base/database.py
Copyright 2025 by Inc, All Rights Reserved. 
2025-03-25 20:06:43
"""

from manifast import *
from collections import namedtuple
from scipy.spatial.transform import Rotation as R
# import mayavi.mlab
import rospy
import sensor_msgs.point_cloud2 as pc2
from autoware_msgs.msg import DetectedObject, DetectedObjectArray
from geometry_msgs.msg import PolygonStamped, Point32
from sensor_msgs.msg import PointCloud2, PointField
from shapely.geometry import Point, Polygon

fields = [
    PointField('x', 0, PointField.FLOAT32, 1),
    PointField('y', 4, PointField.FLOAT32, 1),
    PointField('z', 8, PointField.FLOAT32, 1),
    PointField('rgb', 12, PointField.UINT32, 1),
]
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
        self._load_timestamps(sweep)

    def _load_timestamps(self, sweep):
        clip_info_path = osp.join(
            self.data_dir, 'scene_info.json')
        json_data = read_json_data(clip_info_path)
        self.clip_stamps = {}
        for clip, value in json_data.items():
            frames = []
            frames.extend(value['samples'])
            if sweep:
                frames.extend(value['sweeps'])
            frames = sorted(frames)
            self.clip_stamps[clip] = frames

    def load_calib(self, scene_name, timestamp):
        data = {}
        filepath = os.path.join(
            self.data_dir, f'{scene_name}/calib/{timestamp}.txt')
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
        # data['T_lidar_odom'] = self.load_egopose_tf(scene_name, timestamp)
        data['T_lidar_odom'] = self.load_odompose_tf(scene_name, timestamp)

        return namedtuple('CalibData', data.keys())(*data.values())

    def load_lidar(self, scene_name, lidar_stamp, view=False):
        lidar_path = osp.join(
            self.data_dir, scene_name, 'lidar_1', lidar_stamp+'.bin')
        pointcloud = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 4)
        pointcloud = pointcloud[pointcloud[:, 0] < 110, :]
        return pointcloud

    def load_egopose_tf(self, scene_name, stamp):
        location_path = os.path.join(
            self.data_dir, scene_name, f'pose_ego/{stamp}.txt')

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

    def load_odompose_tf(self, scene_name, stamp):
        location_path = os.path.join(
            self.data_dir, scene_name, f'pose_odom/{stamp}.txt')

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

    def load_label3d(self, scene_name, stamp):
        obs_path = f'{self.data_dir}/{scene_name}/lidar_1_label_3d/{stamp}.txt'
        obs_array = []
        if osp.exists(obs_path):
            obs_array = read_txt_data(obs_path)
        obs_out = []
        for obs in obs_array:
            obs_info = obs.split(' ')
            label = obs_info[0]
            h, w, l = float(obs_info[8]), float(
                obs_info[9]), float(obs_info[10])
            x, y, z = float(obs_info[11]), float(
                obs_info[12]), float(obs_info[13])
            # orient = float(obs_info[3])
            orient = float(obs_info[14])
            token = obs_info[-1]
            obs_i = [token, label, x, y, z, w, h, l, orient]
            obs_out.append(obs_i)
        return obs_out

    def load_label_4d(self, scene_name, stamps):
        obss = []
        for stamp in stamps[::-1]:
            obs_path = f'{self.data_dir}/{scene_name}/label_4d/{stamp}.json'
            if osp.exists(obs_path):
                obs = read_json_data(obs_path)['obstacles']
                if len(obs) == 0:
                    continue
                obss.extend(obs)
            if len(obss) > 0:
                break
        return obss

    def build_semantic_map(self, scene_name):
        map_points = None
        stamps = self.clip_stamps[scene_name]
        for stamp in stamps:
            image, segmap_rgb, pointcloud_rgb, pointcloud_seg_rgb = self.build_semi_cloud(
                scene_name, stamp)
            map_pts = pointcloud_seg_rgb
            map_pts = self.transform_pc(
                map_pts, self.load_calib(scene_name, stamp).T_lidar_odom)
            if map_points is None:
                map_points = map_pts
            else:
                map_points = np.concatenate(
                    (map_points, map_pts), axis=0)
        map_points = map_points.astype(np.float32)
        return map_points

    def build_static_obs_cloud(self, cloud_map, static_obs):
        obs_cloud = None
        for obs in static_obs:
            obs['obj_type'] = obs['obj_type'].lower()
            if 'rut' in obs['obj_type']:
                color = palette[14]
            elif 'puddle' in obs['obj_type']:
                color = palette[12]
            elif 'slit' in obs['obj_type']:
                color = palette[13]
            elif 'building' in obs['obj_type']:
                color = palette[15]
            elif 'warningsign' in obs['obj_type']:
                color = palette[7]
            elif 'signboard' in obs['obj_type']:
                color = palette[8]
            if 'bbox' not in obs['obj_type']:
                polygon_points = np.array(
                    obs['polygons']).reshape(-1, 3)[:, :2]
                self.centroid = np.mean(polygon_points, axis=0)
                sorted_points_clockwise = sorted(
                    polygon_points, key=self.calculate_angle, reverse=True)
                polygon = Polygon(sorted_points_clockwise)
                if polygon.area < 0.5:
                    continue
                points_in_poly = []
                points_out_poly = []
                for point in cloud_map:
                    p = Point(point[0], point[1])
                    if polygon.contains(p):
                        point[3:] = color
                        points_in_poly.append(point)
                    else:
                        points_out_poly.append(point)
                filtered_points = np.array(points_in_poly)
                if len(filtered_points) < 1:
                    continue
                cloud_map = np.array(points_out_poly)
                if obs_cloud is None:
                    obs_cloud = filtered_points
                else:
                    obs_cloud = np.concatenate(
                        (obs_cloud, filtered_points), axis=0)
            else:
                bbox_center = np.array(obs['bbox'][:3])
                w, h, l, yaw = obs['bbox'][3:]
                if 'signboard' in obs['obj_type']:
                    w, h, l = 3, 6, 3
                elif 'warningsign' in obs['obj_type']:
                    w,  l = 1, 1
                xyz_points = cloud_map[:, :3]
                translated_points = xyz_points - bbox_center
                rotation_matrix = np.array([
                    [np.cos(-yaw), -np.sin(-yaw), 0],
                    [np.sin(-yaw),  np.cos(-yaw), 0],
                    [0, 0, 1]
                ])
                rotated_points = np.dot(translated_points, rotation_matrix.T)
                x_min, x_max = -l / 2, l / 2
                y_min, y_max = -w / 2, w / 2
                z_min, z_max = -h / 2, h / 2
                inside_mask = (
                    (rotated_points[:, 0] >= x_min) & (rotated_points[:, 0] <= x_max) &
                    (rotated_points[:, 1] >= y_min) & (rotated_points[:, 1] <= y_max) &
                    (rotated_points[:, 2] >= z_min) & (
                        rotated_points[:, 2] <= z_max)
                )

                points_in_box = cloud_map[inside_mask]
                cloud_map = cloud_map[~inside_mask]

                cloud_map = np.array(cloud_map)
                points_in_poly = []
                points_out_poly = []
                for point in points_in_box:
                    point[3:] = color
                    points_in_poly.append(point)
                filtered_points = np.array(points_in_poly)
                if len(filtered_points) < 1:
                    continue
                if obs_cloud is None:
                    obs_cloud = filtered_points
                else:
                    obs_cloud = np.concatenate(
                        (obs_cloud, filtered_points), axis=0)
        if obs_cloud is None:
            obs_cloud = np.array([[0, 0, 0, 0, 0, 0]])
        non_obs_cloud = cloud_map
        return obs_cloud, non_obs_cloud

    def split_cloud_with_dyna_static(self, pointcloud, dynamic_obss, view=False):
        obs_cloud = np.array([[0, 0, 0, 0, 0, 0]])
        for obs in dynamic_obss:
            bbox_center = np.array(obs[2:5])
            w, h, l, yaw = obs[5:]
            xyz_points = pointcloud[:, :3]
            translated_points = xyz_points - bbox_center
            rotation_matrix = np.array([
                [np.cos(-yaw), -np.sin(-yaw), 0],
                [np.sin(-yaw),  np.cos(-yaw), 0],
                [0, 0, 1]
            ])
            rotated_points = np.dot(translated_points, rotation_matrix.T)
            x_min, x_max = -l / 2, l / 2
            y_min, y_max = -w / 2, w / 2
            z_min, z_max = -h / 2, h / 2
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
            pointcloud = np.array(pointcloud)

        return obs_cloud, pointcloud

    def filter_connected_components4wall_road(self, wall_pc, road_pc):
        road_wall_pc = np.concatenate([wall_pc, road_pc])
        x_max = np.max(road_wall_pc[:, 1])
        x_min = np.min(road_wall_pc[:, 1])
        y_max = np.max(road_wall_pc[:, 0])
        y_min = np.min(road_wall_pc[:, 0])
        scale = 1.
        img_size_x = int((x_max - x_min) / scale)
        img_size_y = int((y_max - y_min) / scale)
        img_road = np.zeros((img_size_x, img_size_y), dtype=np.uint8)
        img_height = np.zeros((img_size_x, img_size_y), dtype=np.uint32)
        img_wall = np.zeros((img_size_x, img_size_y), dtype=np.uint8)
        for pt in road_pc:
            x = int((pt[1] - x_min) / scale)
            y = img_size_y-int((pt[0] - y_min) / scale)
            if 0 <= x < img_size_x and 0 <= y < img_size_y and (img_height[x, y] == 0 or int(pt[2]*10) < img_height[x, y]):
                img_road[x, y] = 255
                img_height[x, y] = int(pt[2]*10)
        for pt in wall_pc:
            x = int((pt[1] - x_min) / scale)
            y = img_size_y-int((pt[0] - y_min) / scale)
            # 保证 x 和 y 不会超出边界
            if 0 <= x < img_size_x and 0 <= y < img_size_y and (img_height[x, y] == 0 or int(pt[2]*10) > img_height[x, y]+0.3):
                img_wall[x, y] = 255
                img_height[x, y] = int(pt[2]*10)

        kernel = np.ones((2, 2), np.uint8)
        img_wall_eroded = cv2.erode(img_wall, kernel, iterations=1)
        img_wall_dilated = cv2.dilate(img_wall_eroded, kernel, iterations=1)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            img_wall, connectivity=8)
        filtered_img_wall = np.zeros_like(img_wall)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= 10:
                mask = (labels == i)
                xs, ys = np.where(mask)
                for x, y in zip(xs, ys):
                    if img_wall_dilated[x, y] == 255:
                        filtered_img_wall[labels == i] = 255
                        break
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            img_road, connectivity=8)
        filtered_img_road = np.zeros_like(img_road)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= 10:
                filtered_img_road[labels == i] = 255
        filtered_img = np.zeros_like(img_road)
        filtered_img[filtered_img_road == 255] = 128
        filtered_img[filtered_img_wall == 255] = 255
        wall_road_cloud = []
        for pt in road_wall_pc:
            x = int((pt[1] - x_min) / scale)
            y = img_size_y-int((pt[0] - y_min) / scale)
            if 0 <= x < img_size_x and 0 <= y < img_size_y:
                if filtered_img[x, y] == 255:
                    pt[3:] = (128, 128, 128)
                    wall_road_cloud.append(pt)
                elif filtered_img[x, y] == 128:
                    pt[3:] = (0, 128, 200)
                    wall_road_cloud.append(pt)

        return np.array(wall_road_cloud), filtered_img, x_min, y_min, scale, img_size_x, img_size_y, filtered_img

    def load_static_pc(self, scene_name, lidar_stamp, view=False):
        lidar_path = osp.join(
            self.data_dir, scene_name, 'lidar_1', lidar_stamp+'.bin')
        pointcloud = np.fromfile(
            lidar_path, dtype=np.float32).reshape(-1, 4)
        pointcloud = pointcloud[pointcloud[:, 0] < 110, :]
        dynamic_obss = self.load_label3d(scene_name, lidar_stamp)
        for obs in dynamic_obss:
            bbox_center = np.array(obs[2:5])
            w, h, l, yaw = obs[5:]
            xyz_points = pointcloud[:, :3]
            translated_points = xyz_points - bbox_center
            rotation_matrix = np.array([
                [np.cos(-yaw), -np.sin(-yaw), 0],
                [np.sin(-yaw),  np.cos(-yaw), 0],
                [0, 0, 1]
            ])
            rotated_points = np.dot(translated_points, rotation_matrix.T)
            x_min, x_max = -l / 2, l / 2
            y_min, y_max = -w / 2, w / 2
            z_min, z_max = -h / 2, h / 2
            inside_mask = (
                (rotated_points[:, 0] >= x_min) & (rotated_points[:, 0] <= x_max) &
                (rotated_points[:, 1] >= y_min) & (rotated_points[:, 1] <= y_max) &
                (rotated_points[:, 2] >= z_min) & (
                    rotated_points[:, 2] <= z_max)
            )
            points_in_box = pointcloud[inside_mask]
            pointcloud = pointcloud[~inside_mask]
            pointcloud = np.array(pointcloud)

        return pointcloud

    def transform_pc(self, pointcloud, Tr_velo_to_imu):
        points_in_velo = np.hstack(
            (pointcloud[:, :3], np.ones((pointcloud.shape[0], 1))))
        points_in_velo_hom = np.hstack(
            (points_in_velo[:, :3], np.ones((points_in_velo.shape[0], 1))))
        points_in_imu = Tr_velo_to_imu.dot(points_in_velo_hom.T).T
        pointcloud_imu = np.hstack(
            (points_in_imu[:, :3], pointcloud[:, 3:]))

        return pointcloud_imu

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
                cv2.circle(image_pt, (u, v), 1, (0, 255, 0), -1)
                r, g, b = segmap_rgb[v, u, :]
                if (r, g, b) == (0, 0, 0):
                    pointcloud_valid[i] = False

            elif not enable_outline:
                pointcloud_valid[i] = False
            else:
                colors_seg[i] = (200, 200, 200)
        pointcloud_rgb = np.hstack((pointcloud_in, colors))[pointcloud_valid]
        pointcloud_seg_rgb = np.hstack((pointcloud_in, colors_seg))[
            pointcloud_valid]
        return image_pt, pointcloud_rgb, pointcloud_seg_rgb

    def build_semi_cloud(self, scene_name, stamp, pad_cloud=[]):
        calib = self.load_calib(scene_name, stamp)
        points = self.load_static_pc(scene_name, stamp)[:, :3]
        if len(pad_cloud) > 0:
            points = np.concatenate(
                (points, pad_cloud), axis=0)
        image_path = osp.join(
            self.data_dir, scene_name, 'camera_1', f'{stamp}.jpg')
        segmap_path = osp.join(
            self.data_dir, scene_name, 'camera_1_label_segmap', f'{stamp}.png')
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

    def distances_filter(self, pts_3d, center_pt):
        pts_3d_out = []
        for pt in pts_3d:
            dist = np.sqrt((pt[0]-center_pt[0])**2 + (pt[1]-center_pt[1])**2)
            if dist < 1.:
                pts_3d_out.append(pt)
        return pts_3d_out

    def extra_obs_cloud(self, pc_seg_rgb, scene_name, stamp, image):
        obs_array = DetectedObjectArray()
        warn_pc = pc_seg_rgb[np.all(pc_seg_rgb[:, 3:] == (128, 0, 64), axis=1)]
        sign_pc = pc_seg_rgb[np.all(
            pc_seg_rgb[:, 3:] == (128, 0, 128), axis=1)]
        fence_pc = pc_seg_rgb[np.all(
            pc_seg_rgb[:, 3:] == (64, 0, 128), axis=1)]
        rock_pc = pc_seg_rgb[np.all(
            pc_seg_rgb[:, 3:] == (255, 128, 0), axis=1)]
        pointcloud = np.vstack((warn_pc, sign_pc, fence_pc, rock_pc))
        if len(pointcloud) < 3:
            return obs_array
        calib_param = self.load_calib(scene_name, stamp)
        T_odom_lidar = np.linalg.inv(calib_param.T_lidar_odom)
        Tr_velo_to_cam = calib_param.T_lidar_camera.dot(T_odom_lidar)
        P2 = calib_param.P2
        velo_points_hom = np.hstack(
            (pointcloud[:, :3], np.ones((pointcloud.shape[0], 1))))
        cam_points = Tr_velo_to_cam.dot(velo_points_hom.T).T
        img_points_hom = P2.dot(cam_points.T).T
        img_points = img_points_hom[:, :2] / img_points_hom[:, 2, np.newaxis]
        img_h, img_w, _ = image.shape

        label_2d_path = osp.join(
            self.data_dir, scene_name, 'camera_1_label_2d', f'{stamp}.json')
        label_2d = read_json_data(label_2d_path)
        for obs in label_2d['instance']:
            if obs['obj_type'] not in ['Warningsign', 'Signboard',  'Fence', 'Rock']:
                continue
            x, y, w, h = obs['bbox'][0]*img_w, obs['bbox'][1] * \
                img_h, obs['bbox'][2]*img_w, obs['bbox'][3]*img_h
            x1, y1, x2, y2 = x-w/2, y-h/2, x+w/2, y+h/2
            pts_3d = []
            center_pt = pointcloud[0]
            min_dist = 10
            for i, point in enumerate(img_points):
                u, v = int(point[0]), int(point[1])
                if x1 <= u <= x2 and y1 <= v <= y2:
                    dist = np.sqrt((u-x)**2+(v-y)**2)
                    if dist < min_dist:
                        min_dist = dist
                        center_pt = pointcloud[i]
                    pts_3d.append(pointcloud[i])
                    cv2.circle(image, (u, v), 3, (0, 255, 0), -1)
            if min_dist > 6:
                continue
            polygon = PolygonStamped()
            pointcloud_data = []
            pts_3d = self.distances_filter(pts_3d, center_pt)
            if len(pts_3d) < 3:
                continue
            for pt in pts_3d:
                polygon.polygon.points.append(Point32(pt[0], pt[1], pt[2]))
                x, y, z = pt[:3]
                r, g, b = int(pt[3]), int(pt[4]), int(pt[5])
                rgb = (int(r) << 16) | (int(g) << 8) | int(b)
                pointcloud_data.append((x, y, z, rgb))
            obs_det = DetectedObject()
            obs_det.label = obs['obj_type']+"_bbox"
            obs_det.valid = True
            obs_det.pose_reliable = True
            msg = PointCloud2()
            msg.header.frame_id = "world_link"
            msg.header.stamp = rospy.Time.from_seconds(float(stamp))
            pointcloud_msg = pc2.create_cloud(
                msg.header, fields, pointcloud_data)
            obs_det.pointcloud = pointcloud_msg
            obs_array.objects.append(obs_det)
        return obs_array

    def calculate_angle(self, point):
        return np.arctan2(point[1] - self.centroid[1], point[0] - self.centroid[0])
