#!/usr/bin/env python3
# coding=utf-8
"""
brief: 
Version: v0.0.1
Author: knightdby  && knightdby@163.com
Date: 2024-09-06 16:02:11
Description: 
LastEditors: knightdby
LastEditTime: 2025-05-19 14:49:02
FilePath: /UnScenes3D/pipline/uns_label4d/src/obs_4d_builder.py
Copyright 2025 by Inc, All Rights Reserved. 
2024-09-06 16:02:11
"""

from manifast import *
import time
from pipline.uns_label4d.base.database import Database
import numpy as np
from autoware_msgs.msg import DetectedObjectArray
from cv_bridge import CvBridge
from sensor_msgs.msg import Image as ros_image
import signal
import tf
import rospy
from sensor_msgs.msg import PointCloud2, PointField
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point, Twist
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

    rate = rospy.Rate(1)
    signal.signal(signal.SIGINT, signal_handler)

    db = Database('./data/raw_data',
                  sweep=True,
                  )
    for clip_idx in tqdm(range(0, len(db.clip_stamps.keys()))):
        save_path = None
        clip_name = list(db.clip_stamps.keys())[clip_idx]
        stamps = db.clip_stamps[clip_name]
        print(clip_name, stamps[0], stamps[-1])
        rospy.set_param('/local_map_pub/save_dir2hard_disk',
                        os.path.abspath(f'./data/raw_data/{clip_name}'))
        for stamp_idx in range(len(stamps)):
            stamp = stamps[stamp_idx]
            map_points = None
            for i in [2, 4, 6]:
                stmp = stamps[min(stamp_idx+i, len(stamps)-1)]
                pc_veh = db.load_static_pc(clip_name, stmp)[:, :3]
                pc_map = db.transform_pc(
                    pc_veh, db.load_calib(clip_name, stmp).T_lidar_odom)
                if map_points is None:
                    map_points = pc_map
                else:
                    map_points = np.concatenate(
                        (map_points, pc_map), axis=0)
            T_odom_lidar = np.linalg.inv(
                db.load_calib(clip_name, stamp).T_lidar_odom)
            pad_cloud = db.transform_pc(
                map_points, T_odom_lidar)
            image, segmap_rgb, pc_rgb, pc_seg_rgb = db.build_semi_cloud(
                clip_name, stamp, pad_cloud)

            pointcloud_data_veh = []
            for point in pc_seg_rgb:
                x, y, z = point[:3]
                r, g, b = int(point[3]), int(point[4]), int(point[5])
                rgb = (int(r) << 16) | (int(g) << 8) | int(b)
                pointcloud_data_veh.append((x, y, z, rgb))
            pc_seg_rgb = db.transform_pc(
                pc_seg_rgb, db.load_calib(clip_name, stamp).T_lidar_odom)
            msg = PointCloud2()
            msg.header.frame_id = "world_link"
            msg.header.stamp = rospy.Time.from_seconds(float(stamp))
            pointcloud_data = []
            for point in pc_seg_rgb:
                x, y, z = point[:3]
                r, g, b = int(point[3]), int(point[4]), int(point[5])
                rgb = (int(r) << 16) | (int(g) << 8) | int(b)
                pointcloud_data.append((x, y, z, rgb))
            obs_array = db.extra_obs_cloud(pc_seg_rgb, clip_name, stamp, image)
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

            calib = db.load_calib(clip_name, stamp)
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
            trans_l2m = db.load_odompose_tf(clip_name, stamp)
            x, y, z = trans_l2m[0:3, 3]
            quaternion = tf.transformations.quaternion_from_matrix(
                trans_l2m)
            odom.pose.pose.position = Point(x, y, z)
            # odom.pose.pose.orientation = quaternion
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
