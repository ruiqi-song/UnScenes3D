#!/usr/bin/env python3
# coding=utf-8
"""
brief: 
Version: v0.0.1
Author: knightdby  && knightdby@163.com
Date: 2024-09-06 16:02:11
Description: 
LastEditors: knightdby
LastEditTime: 2025-05-16 09:22:54
FilePath: /UnScenes3D/pipline/uns_label4d/src/static_obs_server.py
Copyright 2025 by Inc, All Rights Reserved. 
2024-09-06 16:02:11
"""


import time
import pickle
import mayavi.mlab
import numpy as np
from sklearn.cluster import DBSCAN
from scipy.spatial import KDTree
from autoware_msgs.msg import DetectedObject, DetectedObjectArray
from geometry_msgs.msg import PolygonStamped, Point32
from cv_bridge import CvBridge
from sensor_msgs.msg import Image as ros_image
import signal
import tf
import rospy
from sensor_msgs.msg import PointCloud2, PointField
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Quaternion, Point, Pose, Twist
from manifast import *
from collections import namedtuple
from scipy.spatial.transform import Rotation as R
import open3d as o3d
import sensor_msgs.point_cloud2 as pc2


classes_2d_seg = ['background', 'truck', 'widebody', 'car',
                  'excavator', 'machinery', 'person',
                  'warningsign', 'signboard', 'fence',
                  'cable', 'rock', 'puddle',
                  'slit', 'rut', 'building',
                  'retainingwall', 'road']
# RGB(0, 0, 0), RGB(0, 100, 128), RGB(0, 150, 128), RGB(0, 200, 128)
# RGB(0, 255, 128), RGB(0, 255, 255), RGB(128, 0, 0),
# RGB(128, 0, 64), RGB(128, 0, 128), RGB(64, 0, 128),
# RGB(128, 0, 255), RGB(255, 128, 0), RGB(0, 0, 255),
# RGB(64, 64, 0), RGB(255,  255, 0), RGB(128, 64, 128),
# RGB(128, 128, 128), RGB(0, 128, 200),
palette = [(0, 0, 0), (0, 100, 128), (0, 150, 128), (0, 200, 128),
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
            self.data_dir, 'infos/data.json')
        json_data = read_json_data(clip_info_path)
        self.clip_info = {}
        for clip, value in json_data.items():
            frames = []
            frames.extend(value['samples'])
            if sweep:
                frames.extend(value['sweeps'])
            frames = sorted(frames)
            self.clip_info[clip] = frames

    def load_dynamic_obs(self, stamp):
        obs_path = f'/media/knight/disk2knight/htmine_occ/semi_data/dynamic_obs/{stamp}.txt'
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
            obs_i = [label, x, y, z, w, h, l, orient]
            obs_out.append(obs_i)
        return obs_out

    def load_lidar(self, lidar_stamp, view=False):
        lidar_path = osp.join(
            self.data_dir, 'sample/clouds', lidar_stamp+'.bin')
        if not osp.exists(lidar_path):
            lidar_path = lidar_path.replace('sample', 'sweep')
        pointcloud = np.fromfile(lidar_path, dtype=np.float32)
        pointcloud = pointcloud.reshape(-1, 4)
        dynamic_obss = self.load_dynamic_obs(lidar_stamp)
        for obs in dynamic_obss:
            bbox_center = np.array(obs[1:4])
            w, h, l, yaw = obs[4:]
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

    def load_egopose(self, stamp):
        location_path = osp.join(
            self.data_dir, 'sample/ego_pose', stamp+'.txt')
        if not osp.exists(location_path):
            location_path = location_path.replace('sample', 'sweep')
        lines = []

        with open(location_path, 'r') as file_to_read:
            while True:
                line = file_to_read.readline()  # 整行读取数据
                lines.append(line)
                if not line:
                    break
        key, value = lines[0].split(' ', 1)
        return np.array([float(x) for x in value.split()])

    def load_calib(self, timestamp):
        data = {}
        filepath = os.path.join(
            self.data_dir, f'sample/calibs/{timestamp}.txt')
        if not osp.exists(filepath):
            filepath = filepath.replace('sample', 'sweep')
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

        return namedtuple('CalibData', data.keys())(*data.values())

    def transform_pointcloud(self, pointcloud, Tr_velo_to_imu):
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

    def build_semi_cloud(self, stamp, pad_cloud=[]):
        calib = self.load_calib(stamp)
        points = self.load_lidar(stamp)[:, :3]
        if len(pad_cloud) > 0:
            points = np.concatenate(
                (points, pad_cloud), axis=0)
        image_path = osp.join(
            self.data_dir, 'sample/images', f'{stamp}.jpg')
        segmap_path = osp.join(
            self.data_dir, 'sample/label_segmap', f'{stamp}.png')
        if not osp.exists(image_path):
            image_path = image_path.replace('sample', 'sweep')
            segmap_path = segmap_path.replace('sample', 'sweep')
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

    def extra_obs_cloud(self, pc_seg_rgb, stamp, image):
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
        calib_param = db.load_calib(stamp)
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
            self.data_dir, 'sample/label_2d', f'{stamp}.json')
        if not osp.exists(label_2d_path):
            label_2d_path = label_2d_path.replace('sample', 'sweep')
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


def signal_handler(sig, frame):
    sys.exit(0)


fields = [
    PointField('x', 0, PointField.FLOAT32, 1),
    PointField('y', 4, PointField.FLOAT32, 1),
    PointField('z', 8, PointField.FLOAT32, 1),
    PointField('rgb', 12, PointField.UINT32, 1),
]
fields_intens = [
    PointField('x', 0, PointField.FLOAT32, 1),
    PointField('y', 4, PointField.FLOAT32, 1),
    PointField('z', 8, PointField.FLOAT32, 1),
    PointField('intensity', 12, PointField.FLOAT32, 1),
]

if __name__ == "__main__":

    rospy.init_node('local_map_publisher', anonymous=True)
    br = tf.TransformBroadcaster()
    seg_cloud_pub = rospy.Publisher(
        '/local_map_pub/seg_cloud', PointCloud2, queue_size=1)
    seg_cloud_veh_pub = rospy.Publisher(
        '/local_map_pub/seg_cloud_veh', PointCloud2, queue_size=1)
    image_pub = rospy.Publisher(
        '/local_map_pub/image', ros_image, queue_size=1)
    odom_pub = rospy.Publisher('/local_map_pub/odom', Odometry, queue_size=10)
    cam_bbox_pub = rospy.Publisher(
        '/local_map_pub/obs_bbox_2d',  DetectedObjectArray, queue_size=1)
    rospy.set_param('/local_map_pub/save_dir2hard_disk',
                    '/media/knight/disk2knight/htmine_occ/semi_data')
    rate = rospy.Rate(1)
    signal.signal(signal.SIGINT, signal_handler)

    db = Database('/media/knight/disk2knight/htmine_occ/HTMINE_Occ',
                  #   sweep=True)
                  sweep=False)

    for clip in tqdm(list(db.clip_info.keys())[0:]):
        stamps = sorted(db.clip_info[clip])
        print(clip, stamps[0], stamps[-1])
        for stamp_idx in range(len(stamps)):
            stamp = stamps[stamp_idx]
            map_points = None
            for i in [2, 4, 6]:
                stmp = stamps[min(stamp_idx+i, len(stamps)-1)]
                pc_veh = db.load_lidar(stmp)[:, :3]
                pc_map = db.transform_pointcloud(
                    pc_veh, db.load_calib(stmp).T_lidar_odom)
                if map_points is None:
                    map_points = pc_map
                else:
                    map_points = np.concatenate(
                        (map_points, pc_map), axis=0)
            T_odom_lidar = np.linalg.inv(db.load_calib(stamp).T_lidar_odom)
            pad_cloud = db.transform_pointcloud(
                map_points, T_odom_lidar)
            image, segmap_rgb, pc_rgb, pc_seg_rgb = db.build_semi_cloud(
                stamp, pad_cloud)

            pointcloud_data_veh = []
            for point in pc_seg_rgb:
                x, y, z = point[:3]
                r, g, b = int(point[3]), int(point[4]), int(point[5])
                rgb = (int(r) << 16) | (int(g) << 8) | int(b)
                pointcloud_data_veh.append((x, y, z, rgb))
            pc_seg_rgb = db.transform_pointcloud(
                pc_seg_rgb, db.load_calib(stamp).T_lidar_odom)
            msg = PointCloud2()
            msg.header.frame_id = "world_link"
            msg.header.stamp = rospy.Time.from_seconds(float(stamp))
            pointcloud_data = []
            for point in pc_seg_rgb:
                x, y, z = point[:3]
                r, g, b = int(point[3]), int(point[4]), int(point[5])
                rgb = (int(r) << 16) | (int(g) << 8) | int(b)
                pointcloud_data.append((x, y, z, rgb))
            obs_array = db.extra_obs_cloud(pc_seg_rgb, stamp, image)
            obs_array.header.stamp = rospy.Time.from_seconds(float(stamp))
            obs_array.header.frame_id = "world_link"
            cam_bbox_pub.publish(obs_array)
            pointcloud_seg_rgb_msg = pc2.create_cloud(
                msg.header, fields, pointcloud_data)
            seg_cloud_pub.publish(pointcloud_seg_rgb_msg)
            msg.header.frame_id = "vehicle_link"
            pointcloud_seg_rgb_veh_msg = pc2.create_cloud(
                msg.header, fields, pointcloud_data_veh)
            seg_cloud_veh_pub.publish(pointcloud_seg_rgb_veh_msg)
            image = np.vstack((image, segmap_rgb[:, :, ::-1]))
            bridge = CvBridge()
            image_msg = bridge.cv2_to_imgmsg(image, encoding="bgr8")
            image_msg.header.stamp = rospy.Time.from_seconds(float(stamp))
            image_pub.publish(image_msg)

            calib = db.load_calib(stamp)
            Tr_velo_to_imu = calib.T_lidar_odom
            rotation_matrix = Tr_velo_to_imu[0:3, 0:3]
            translation = Tr_velo_to_imu[0:3, 3]
            quaternion = tf.transformations.quaternion_from_matrix(
                Tr_velo_to_imu)
            br.sendTransform(
                translation,
                quaternion,
                rospy.Time.from_seconds(float(stamp)),
                "vehicle_link",
                "world_link"
            )

            odom = Odometry()
            odom.header.stamp = rospy.Time.from_seconds(float(stamp))
            odom.header.frame_id = "odom"
            rt_param = db.load_egopose(stamp)
            qx, qy, qz, qw = rt_param[3:]
            x, y, z = rt_param[:3]
            odom.pose.pose.position = Point(x, y, z)
            odom.pose.pose.orientation = Quaternion(qx, qy, qz, qw)
            odom.twist.twist.linear = Twist().linear
            odom.twist.twist.linear.x = 0
            odom.twist.twist.linear.y = 0
            odom.twist.twist.linear.z = 0
            odom.twist.twist.angular.x = 0
            odom.twist.twist.angular.y = 0
            odom.twist.twist.angular.z = 0
            odom_pub.publish(odom)
            if stamp_idx == 0:
                time.sleep(3)

            rate.sleep()

        # break
