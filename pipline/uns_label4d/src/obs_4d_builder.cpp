

/***
 * @brief:
 * @Version: v0.0.1
 * @Author: knightdby  && knightdby@163.com
 * @Date: 2024-09-10 10:27:22
 * @Description:
 * @LastEditors: knightdby
 * @LastEditTime: 2025-05-16 09:23:12
 * @FilePath: /UnScenes3D/pipline/uns_label4d/src/static_obs_builder.cpp
 * @Copyright 2025 by Inc, All Rights Reserved.
 * @2024-09-10 10:27:22
 */

#include <memory>
#include <utility>
#include <vector>
#include <unordered_map>
#include <iostream>
#include <fstream>
#include <ros/ros.h>
#include <yaml-cpp/yaml.h>
#include <nlohmann/json.hpp>

#include <nav_msgs/Odometry.h>
#include <nav_msgs/OccupancyGrid.h>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/time_synchronizer.h>
#include <tf/transform_listener.h>

#include <sensor_msgs/PointCloud2.h>

#include <pcl/common/common.h>
#include <pcl/io/io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/crop_box.h>
#include <pcl/filters/approximate_voxel_grid.h>
#include <pcl/filters/statistical_outlier_removal.h>

#include <pcl_ros/transforms.h>
#include <pcl_ros/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_types_conversion.h>
#include <pcl/point_types.h>

#include <pcl/segmentation/extract_clusters.h>
#include "pcl/search/pcl_search.h"
#include <pcl/filters/passthrough.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/filters/radius_outlier_removal.h>

#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>
#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>

#include <jsk_recognition_msgs/PolygonArray.h>
#include <jsk_recognition_msgs/BoundingBox.h>
#include <jsk_recognition_msgs/BoundingBoxArray.h>
#include <autoware_msgs/DetectedObject.h>
#include <autoware_msgs/DetectedObjectArray.h>
#include <nav_msgs/Odometry.h>

#include <Eigen/Eigen>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <Eigen/Core>

#include "cluster_polygon.h"
#include "polyiou.hpp"
#include "octree_connection.hpp"

#include <autoware_msgs/DetectedObject.h>
#include <geometry_msgs/Point32.h>
#include <algorithm>
#include <vector>
#include <cmath>
#define __APP_NAME__ "StaticObstacleBuilder"

class StaticObstacleBuilder
{
private:
    ros::NodeHandle nh_;
    ros::NodeHandle pnh_;
    ros::Subscriber sub_points_, sub_odom_, sub_obs2d_;
    ros::Publisher mPubLaneSemanticCloud, mPubObsArray, mPubEdgeCloud, mPubMapCloud, mPubCrubCloud, mPubMapImage;

    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::PointCloud2, nav_msgs::Odometry, autoware_msgs::DetectedObjectArray> sync_subs;
    enum class ObsCate
    {
        RUT = 0,
        PUDDLE = 1,
        SLIT = 2,
        CABLE = 3,
        BUILDING = 4,
        ROCK = 5,
        SIGN = 6,
        WALL = 7,
    };
    std_msgs::Header mHeader;

    int _cluster_size_min;
    int _cluster_size_max;
    double _clustering_distance;
    autoware_msgs::DetectedObjectArray out_objects_;

    boost::shared_ptr<std::queue<autoware_msgs::DetectedObjectArray>> mpObstacleQueue;
    boost::shared_ptr<std::queue<autoware_msgs::DetectedObjectArray>> mpObstacle2dQueue;
    boost::shared_ptr<std::queue<pcl::PointCloud<pcl::PointXYZRGB>>> mpCloudQueue;

    int obs_fusion_frame_num_;
    geometry_msgs::Pose mVehiclePose, mVehiclePosePrev;
    std::string mSavedir2Harddisk;
    Eigen::Affine3f mLocaltoGlobal = Eigen::Affine3f::Identity();

public:
    StaticObstacleBuilder(const ros::NodeHandle &nh, const ros::NodeHandle &pnh);
    ~StaticObstacleBuilder() {};
    std::vector<cv::Point2f> PointCloudTo2DProjection(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud, const std::string &axis);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr FilterConnectedComponents(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud, const std::string &axis, int min_size);

    void PointcloudCallback(const sensor_msgs::PointCloud2::ConstPtr &msg,
                            const nav_msgs::Odometry::ConstPtr &gps_msg,
                            const autoware_msgs::DetectedObjectArray::ConstPtr &obj_msg);
    void ExtractObject(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud_global);
    void Euclidean2dClusters(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
                             std::vector<pcl::PointIndices> &ece_inlier,
                             int cluster_size_min);
    autoware_msgs::DetectedObjectArray ExtractClusters(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud_global,
                                                       const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud,
                                                       std::string obs_label);
    geometry_msgs::Point32 computeCentroid(const std::vector<geometry_msgs::Point32> &points);
    void sortConvexHullPoints(autoware_msgs::DetectedObject &detected_object);
    static float computeAngle(const geometry_msgs::Point32 &point, const geometry_msgs::Point32 &centroid);
    void ObsGlobalMerge(autoware_msgs::DetectedObjectArray obs_array);
    void Obs2dCallback(const autoware_msgs::DetectedObjectArray::ConstPtr &input_msg);
    void OdomCallback(const nav_msgs::Odometry::ConstPtr &gps_msg);
    void Procession(void);

    struct OccSemiFilter
    {
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr filter(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &input_cloud, float leaf_size, float leaf_size_z)
        {
            std::unordered_map<std::string, pcl::PointXYZRGB> voxel_map;
            pcl::PointCloud<pcl::PointXYZRGB>::Ptr filtered_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
            for (const auto &point : input_cloud->points)
            {
                int x = static_cast<int>(point.x / leaf_size);
                int y = static_cast<int>(point.y / leaf_size);
                int z = static_cast<int>(point.z / leaf_size_z);
                std::string voxel_key = std::to_string(x) + "_" + std::to_string(y) + "_" + std::to_string(z);
                auto it = voxel_map.find(voxel_key);
                // 无属性，直接赋值
                if (it == voxel_map.end())
                    voxel_map[voxel_key] = point;
                else
                {
                    // road 32968
                    // wall 8421504
                    if (it->second.rgb == 8421504 && point.rgb == 32968)
                    {
                        if (point.z < it->second.z)
                            voxel_map[voxel_key] = point;
                    }
                    else if (it->second.rgb == 32968 && point.rgb == 8421504)
                    {
                        if (point.z < it->second.z)
                            voxel_map[voxel_key] = point;
                    }
                    else if (it->second.rgb > 0 && point.rgb == 8421504)
                        voxel_map[voxel_key] = point;
                    else if (it->second.rgb > 0 && point.rgb == 32968)
                        voxel_map[voxel_key] = point;
                    else
                    {
                        voxel_map[voxel_key] = point;
                    }
                }
            }
            for (const auto &pair : voxel_map)
            {
                filtered_cloud->push_back(pair.second);
            }

            return filtered_cloud;
        }
    };
};

StaticObstacleBuilder::StaticObstacleBuilder(const ros::NodeHandle &nh, const ros::NodeHandle &pnh)
    : nh_(nh), pnh_(pnh)
{
    message_filters::Subscriber<sensor_msgs::PointCloud2> *cloud_sub;
    message_filters::Subscriber<nav_msgs::Odometry> *odom_sub;
    message_filters::Subscriber<autoware_msgs::DetectedObjectArray> *obs_sub;

    message_filters::Synchronizer<sync_subs> *sync;
    cloud_sub = new message_filters::Subscriber<sensor_msgs::PointCloud2>(nh_, "/local_map_pub/seg_cloud", 1000);
    odom_sub = new message_filters::Subscriber<nav_msgs::Odometry>(nh_, "/local_map_pub/odom", 1000);
    obs_sub = new message_filters::Subscriber<autoware_msgs::DetectedObjectArray>(nh_, "/local_map_pub/obs_bbox_2d", 1000);
    sync = new message_filters::Synchronizer<sync_subs>(sync_subs(10),
                                                        *cloud_sub,
                                                        *odom_sub,
                                                        *obs_sub);
    sync->registerCallback(boost::bind(&StaticObstacleBuilder::PointcloudCallback, this, _1, _2, _3));

    mPubLaneSemanticCloud = nh_.advertise<sensor_msgs::PointCloud2>("/camera/segmentor/semanticloud", 1);
    mPubEdgeCloud = nh_.advertise<sensor_msgs::PointCloud2>("/local_map_pub/edge_cloud", 1);
    mPubMapCloud = nh_.advertise<sensor_msgs::PointCloud2>("/local_map_pub/global_cloud", 1);
    mPubCrubCloud = nh_.advertise<sensor_msgs::PointCloud2>("/local_map_pub/crub_cloud", 1);
    mPubObsArray = nh_.advertise<autoware_msgs::DetectedObjectArray>("/local_map_pub/obs_bbox_3d", 1);
    mPubMapImage = nh_.advertise<sensor_msgs::Image>("/local_map_pub/global_map", 1);

    pnh_.param("cluster_size_min", _cluster_size_min, 5);
    ROS_INFO("[%s] cluster_size_min %d", __APP_NAME__, _cluster_size_min);
    pnh_.param("cluster_size_max", _cluster_size_max, 100000);
    ROS_INFO("[%s] cluster_size_max: %d", __APP_NAME__, _cluster_size_max);
    pnh_.param("clustering_distance", _clustering_distance, 0.75);
    ROS_INFO("[%s] clustering_distance: %f", __APP_NAME__, _clustering_distance);

    pnh_.param("obs_fusion_frame_num", obs_fusion_frame_num_, 60);
    ROS_INFO("[%s] obs_fusion_frame_num %d", __APP_NAME__, obs_fusion_frame_num_);

    mpObstacleQueue = boost::make_shared<std::queue<autoware_msgs::DetectedObjectArray>>();
    mpObstacle2dQueue = boost::make_shared<std::queue<autoware_msgs::DetectedObjectArray>>();
    mpCloudQueue = boost::make_shared<std::queue<pcl::PointCloud<pcl::PointXYZRGB>>>();
}

std::vector<cv::Point2f> StaticObstacleBuilder::PointCloudTo2DProjection(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud, const std::string &axis = "z")
{
    std::vector<cv::Point2f> projected_points;
    for (const auto &point : cloud->points)
    {
        if (axis == "z")
        {
            projected_points.emplace_back(point.x, point.y); // 投影到XY平面
        }
        else if (axis == "y")
        {
            projected_points.emplace_back(point.x, point.z); // 投影到XZ平面
        }
        else if (axis == "x")
        {
            projected_points.emplace_back(point.y, point.z); // 投影到YZ平面
        }
    }
    return projected_points;
}
pcl::PointCloud<pcl::PointXYZRGB>::Ptr StaticObstacleBuilder::FilterConnectedComponents(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud, const std::string &axis = "z", int min_size = 100)
{
    std::vector<cv::Point2f> projected_points = PointCloudTo2DProjection(cloud, axis);
    int img_size = 1000;
    cv::Mat img = cv::Mat::zeros(img_size, img_size, CV_8UC1);
    double max_val = 0;
    for (const auto &pt : projected_points)
    {
        if (pt.x > max_val)
            max_val = pt.x;
        if (pt.y > max_val)
            max_val = pt.y;
    }
    double scale = img_size / max_val;
    for (const auto &pt : projected_points)
    {
        int x = static_cast<int>(pt.x * scale);
        int y = static_cast<int>(pt.y * scale);
        img.at<uchar>(y, x) = 255;
    }
    cv::Mat labels, stats, centroids;
    int num_labels = cv::connectedComponentsWithStats(img, labels, stats, centroids);
    cv::Mat filtered_img = cv::Mat::zeros(img_size, img_size, CV_8UC1);
    for (int i = 1; i < num_labels; ++i)
    {
        if (stats.at<int>(i, cv::CC_STAT_AREA) >= min_size)
        {
            filtered_img.setTo(255, labels == i);
        }
    }
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr filtered_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    for (int y = 0; y < filtered_img.rows; ++y)
    {
        for (int x = 0; x < filtered_img.cols; ++x)
        {
            if (filtered_img.at<uchar>(y, x) == 255)
            {
                double px = static_cast<double>(x) / scale;
                double py = static_cast<double>(y) / scale;
                for (const auto &point : cloud->points)
                {
                    if (axis == "z" && std::abs(point.x - px) < 0.001 && std::abs(point.y - py) < 0.001)
                    {
                        filtered_cloud->points.push_back(point);
                        break;
                    }
                    else if (axis == "y" && std::abs(point.x - px) < 0.001 && std::abs(point.z - py) < 0.001)
                    {
                        filtered_cloud->points.push_back(point);
                        break;
                    }
                    else if (axis == "x" && std::abs(point.y - px) < 0.001 && std::abs(point.z - py) < 0.001)
                    {
                        filtered_cloud->points.push_back(point);
                        break;
                    }
                }
            }
        }
    }

    filtered_cloud->width = filtered_cloud->points.size();
    filtered_cloud->height = 1;
    filtered_cloud->is_dense = true;

    return filtered_cloud;
}

void StaticObstacleBuilder::Obs2dCallback(const autoware_msgs::DetectedObjectArray::ConstPtr &input_msg)
{
    autoware_msgs::DetectedObjectArray out_objects;
    for (const auto &in_object : input_msg->objects)
    {
        autoware_msgs::DetectedObject out_object = in_object;
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_ptr(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::fromROSMsg(in_object.pointcloud, *cloud_ptr);
        std::vector<pcl::PointIndices> cluster_indices;
        if (cloud_ptr->size() < 5)
            continue;
        Euclidean2dClusters(cloud_ptr, cluster_indices, 5);

        pcl::PointCloud<pcl::PointXYZ>::Ptr object_cloud(new pcl::PointCloud<pcl::PointXYZ>);
        int cloud_max_num = 0;
        for (auto iter = cluster_indices.begin(); iter != cluster_indices.end(); ++iter)
        {
            pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
            for (auto pit = iter->indices.begin(); pit != iter->indices.end(); ++pit)
                cloud->points.push_back(cloud_ptr->points[*pit]);
            if (cloud->size() > cloud_max_num)
            {
                cloud_max_num = cloud->size();
                object_cloud->clear();
                *object_cloud = *cloud;
            }
        }
        if (cloud_max_num < 5)
            continue;
        pcl::PointXYZ min_point, max_point;
        pcl::getMinMax3D(*object_cloud, min_point, max_point);
        double orientation_angle;
        float length, width, height;

        length = max_point.x - min_point.x;
        width = max_point.y - min_point.y;
        height = max_point.z - min_point.z;

        jsk_recognition_msgs::BoundingBox bounding_box;
        bounding_box.header = input_msg->header;

        bounding_box.pose.position.x = min_point.x + length / 2;
        bounding_box.pose.position.y = min_point.y + width / 2;
        bounding_box.pose.position.z = min_point.z + height / 2;

        bounding_box.dimensions.x = ((length < 0) ? -1 * length : length);
        bounding_box.dimensions.y = ((width < 0) ? -1 * width : width);
        bounding_box.dimensions.z = ((height < 0) ? -1 * height : height);
        geometry_msgs::PolygonStamped polygon;
        double rz = 0;
        {
            std::vector<cv::Point2f> points;
            for (unsigned int i = 0; i < object_cloud->points.size(); i++)
            {
                cv::Point2f pt;
                pt.x = object_cloud->points[i].x;
                pt.y = object_cloud->points[i].y;
                points.push_back(pt);
            }
            std::vector<cv::Point2f> hull;
            cv::convexHull(points, hull);
            polygon.header = input_msg->header;
            for (size_t i = 0; i < hull.size() + 1; i++)
            {
                geometry_msgs::Point32 point;
                point.x = hull[i % hull.size()].x;
                point.y = hull[i % hull.size()].y;
                point.z = max_point.z + 3;
                polygon.polygon.points.push_back(point);
            }

            cv::RotatedRect box = minAreaRect(hull);
            rz = box.angle * 3.14 / 180;
            bounding_box.pose.position.x = box.center.x;
            bounding_box.pose.position.y = box.center.y;
            bounding_box.dimensions.x = box.size.width;
            bounding_box.dimensions.y = box.size.height;
        }
        tf::Quaternion quat = tf::createQuaternionFromRPY(0.0, 0.0, rz);
        tf::quaternionTFToMsg(quat, bounding_box.pose.orientation);
        out_object.pose = bounding_box.pose;
        out_object.dimensions = bounding_box.dimensions;
        out_object.convex_hull = polygon;

        sensor_msgs::PointCloud2 object_cloud_msg;
        pcl::toROSMsg(*object_cloud, object_cloud_msg);
        out_object.pointcloud = object_cloud_msg;
        out_object.pointcloud.header = input_msg->header;
        out_object.valid = true;
        out_object.pose_reliable = true;
        out_objects.objects.push_back(out_object);
    }
    out_objects.header = input_msg->header;
    mpObstacle2dQueue->push(out_objects);
    if (mpObstacle2dQueue->size() > obs_fusion_frame_num_ && mpObstacle2dQueue->size() > 0)
        mpObstacle2dQueue->pop();
}
void StaticObstacleBuilder::OdomCallback(const nav_msgs::Odometry::ConstPtr &gps_msg)
{
    mVehiclePose = gps_msg->pose.pose;

    Eigen::Quaternionf quat(mVehiclePose.orientation.w,
                            mVehiclePose.orientation.x,
                            mVehiclePose.orientation.y,
                            mVehiclePose.orientation.z);
    Eigen::Translation3f translation(mVehiclePose.position.x,
                                     mVehiclePose.position.y,
                                     mVehiclePose.position.z);
    mLocaltoGlobal = translation * quat.toRotationMatrix();

    double shift_distance = sqrt(pow(mVehiclePose.position.x - mVehiclePosePrev.position.x, 2.0) + pow(mVehiclePose.position.y - mVehiclePosePrev.position.y, 2.0));
    if (shift_distance > 50.0)
    {
        while (!mpObstacleQueue->empty())
            mpObstacleQueue->pop();
        while (!mpObstacle2dQueue->empty())
            mpObstacle2dQueue->pop();
        while (!mpCloudQueue->empty())
            mpCloudQueue->pop();
    }
    mVehiclePosePrev = mVehiclePose;
}

geometry_msgs::Point32 StaticObstacleBuilder::computeCentroid(const std::vector<geometry_msgs::Point32> &points)
{
    geometry_msgs::Point32 centroid;
    float x_sum = 0.0;
    float y_sum = 0.0;
    for (const auto &point : points)
    {
        x_sum += point.x;
        y_sum += point.y;
    }
    centroid.x = x_sum / points.size();
    centroid.y = y_sum / points.size();
    return centroid;
}

float StaticObstacleBuilder::computeAngle(const geometry_msgs::Point32 &point, const geometry_msgs::Point32 &centroid)
{
    return std::atan2(point.y - centroid.y, point.x - centroid.x);
}

void StaticObstacleBuilder::sortConvexHullPoints(autoware_msgs::DetectedObject &detected_object)
{
    auto &points = detected_object.convex_hull.polygon.points;

    geometry_msgs::Point32 centroid = computeCentroid(points);

    std::sort(points.begin(), points.end(), [&centroid](const geometry_msgs::Point32 &p1, const geometry_msgs::Point32 &p2)
              { return computeAngle(p1, centroid) < computeAngle(p2, centroid); });
}
void StaticObstacleBuilder::Euclidean2dClusters(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
                                                std::vector<pcl::PointIndices> &ece_inlier,
                                                int cluster_size_min = 5)
{
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_2d(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::copyPointCloud(*cloud, *cloud_2d);
    for (size_t i = 0; i < cloud_2d->points.size(); i++)
        cloud_2d->points[i].z = 0;
    if (cloud_2d->points.size() > 0)
        tree->setInputCloud(cloud_2d);

    pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
    ec.setClusterTolerance(_clustering_distance); //
    ec.setMinClusterSize(cluster_size_min);
    ec.setMaxClusterSize(_cluster_size_max);
    ec.setSearchMethod(tree);
    ec.setInputCloud(cloud_2d);
    ec.extract(ece_inlier);
}

autoware_msgs::DetectedObjectArray StaticObstacleBuilder::ExtractClusters(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud_global,
                                                                          const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud,
                                                                          std::string obs_label = "")
{
    autoware_msgs::DetectedObjectArray obs_array;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_ptr(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::copyPointCloud(*cloud, *cloud_ptr);
    std::vector<pcl::PointIndices> cluster_indices;
    Euclidean2dClusters(cloud_ptr, cluster_indices);
    for (auto iter = cluster_indices.begin(); iter != cluster_indices.end(); ++iter)
    {
        pcl::PointCloud<pcl::PointXYZ>::Ptr object_cloud(new pcl::PointCloud<pcl::PointXYZ>);
        for (auto pit = iter->indices.begin(); pit != iter->indices.end(); ++pit)
            object_cloud->points.push_back(cloud_ptr->points[*pit]);
        if (object_cloud->size() < 5)
            continue;

        pcl::PointXYZ min_point, max_point;
        pcl::getMinMax3D(*object_cloud, min_point, max_point);
        double orientation_angle;
        float length, width, height;
        length = max_point.x - min_point.x;
        width = max_point.y - min_point.y;
        height = max_point.z - min_point.z;

        jsk_recognition_msgs::BoundingBox bounding_box;
        bounding_box.header = mHeader;

        bounding_box.pose.position.x = min_point.x + length / 2;
        bounding_box.pose.position.y = min_point.y + width / 2;
        bounding_box.pose.position.z = min_point.z + height / 2;

        bounding_box.dimensions.x = ((length < 0) ? -1 * length : length);
        bounding_box.dimensions.y = ((width < 0) ? -1 * width : width);
        bounding_box.dimensions.z = ((height < 0) ? -1 * height : height);

        geometry_msgs::PolygonStamped polygon;
        polygon.header = mHeader;

        ClusterPolygon polygon_cluster;
        std::vector<int> convexIndice;
        double concavity{2};
        double lengthThreshold{3};
        pcl::PointCloud<pcl::PointXYZI>::Ptr clusterPtr(new pcl::PointCloud<pcl::PointXYZI>);
        pcl::copyPointCloud(*object_cloud, *clusterPtr);
        pcl::PointCloud<pcl::PointXYZ>::Ptr surface_hull(new pcl::PointCloud<pcl::PointXYZ>);
        if (polygon_cluster.findSmallestPolygon(clusterPtr, convexIndice))
        {

            std::vector<std::array<double, 3>> points(clusterPtr->size());
            for (int j = 0; j < points.size(); j++)
                points[j] = {clusterPtr->points[j].x, clusterPtr->points[j].y, static_cast<double>(j)};
            auto concave_points = concaveman<double, 16>(points, convexIndice, concavity, lengthThreshold);
            for (auto &pt : concave_points)
            {
                pcl::PointXYZI ps = clusterPtr->points[static_cast<int>(pt[2])];
                geometry_msgs::Point32 point;
                point.x = ps.x;
                point.y = ps.y;
                point.z = ps.z;
                polygon.polygon.points.push_back(point);
            }
        }

        double rz = 0;
        {
            std::vector<cv::Point2f> points;
            for (unsigned int i = 0; i < object_cloud->points.size(); i++)
            {
                cv::Point2f pt;
                pt.x = object_cloud->points[i].x;
                pt.y = object_cloud->points[i].y;
                points.push_back(pt);
            }
            std::vector<cv::Point2f> hull;
            cv::convexHull(points, hull);
            for (size_t i = 0; i < hull.size() + 1; i++)
            {
                geometry_msgs::Point32 point;
                point.x = hull[i % hull.size()].x;
                point.y = hull[i % hull.size()].y;
                point.z = max_point.z + 3;
            }
            cv::RotatedRect box = minAreaRect(hull);
            rz = box.angle * 3.14 / 180;
            bounding_box.pose.position.x = box.center.x;
            bounding_box.pose.position.y = box.center.y;
            bounding_box.dimensions.x = box.size.width;
            bounding_box.dimensions.y = box.size.height;
        }
        tf::Quaternion quat = tf::createQuaternionFromRPY(0.0, 0.0, rz);
        tf::quaternionTFToMsg(quat, bounding_box.pose.orientation);
        autoware_msgs::DetectedObject obs;
        obs.pose = bounding_box.pose;
        obs.dimensions = bounding_box.dimensions;
        obs.convex_hull = polygon;

        sensor_msgs::PointCloud2 object_cloud_msg;
        pcl::toROSMsg(*object_cloud, object_cloud_msg);
        obs.pointcloud = object_cloud_msg;
        obs.pointcloud.header = mHeader;
        obs.valid = true;
        obs.pose_reliable = true;
        obs.label = obs_label;
        obs_array.objects.push_back(obs);
    }
    return obs_array;
}
void StaticObstacleBuilder::ExtractObject(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud_global)
{
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr pc_rut(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr pc_puddle(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr pc_slit(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr pc_cable(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr pc_building(new pcl::PointCloud<pcl::PointXYZRGB>);
    for (auto pt : cloud_global->points)
    {
        switch (pt.rgba)
        {
        case 16776960:
            pc_rut->points.push_back(pt);
            break;
        case 255:
            pc_puddle->points.push_back(pt);
            break;
        case 13158400:
            pc_slit->points.push_back(pt);
            break;
        case 8388863:
            pc_cable->points.push_back(pt);
            break;
        case 8405120:
            pc_building->points.push_back(pt);
            break;
        default:
            break;
        }
    }
    autoware_msgs::DetectedObjectArray out_objects;
    autoware_msgs::DetectedObjectArray rut_obs = ExtractClusters(cloud_global, pc_rut, "RUT");
    autoware_msgs::DetectedObjectArray pudding_obs = ExtractClusters(cloud_global, pc_puddle, "PUDDLE");
    autoware_msgs::DetectedObjectArray slit_obs = ExtractClusters(cloud_global, pc_slit, "SLIT");
    autoware_msgs::DetectedObjectArray build_obs = ExtractClusters(cloud_global, pc_building, "BUILDING");

    for (auto obj : rut_obs.objects)
        out_objects.objects.push_back(obj);
    for (auto obj : pudding_obs.objects)
        out_objects.objects.push_back(obj);
    for (auto obj : slit_obs.objects)
        out_objects.objects.push_back(obj);
    for (auto obj : build_obs.objects)
        out_objects.objects.push_back(obj);
    out_objects.header = mHeader;
    mpObstacleQueue->push(out_objects);
    if (mpObstacleQueue->size() > obs_fusion_frame_num_ && mpObstacleQueue->size() > 0)
        mpObstacleQueue->pop();
    autoware_msgs::DetectedObjectArray obs_array;
    std::queue<autoware_msgs::DetectedObjectArray> obs_global_queue(*mpObstacleQueue);
    obs_array.header = mHeader;
    while (obs_global_queue.size())
    {
        autoware_msgs::DetectedObjectArray obs_global_array = obs_global_queue.front();
        for (auto obs : obs_global_array.objects)
            obs_array.objects.push_back(obs);
        obs_global_queue.pop();
    }
    std::queue<autoware_msgs::DetectedObjectArray> obs_global2d_queue(*mpObstacle2dQueue);
    while (obs_global2d_queue.size())
    {
        autoware_msgs::DetectedObjectArray obs_global_array = obs_global2d_queue.front();
        for (auto obs : obs_global_array.objects)
            obs_array.objects.push_back(obs);
        obs_global2d_queue.pop();
    }
    ObsGlobalMerge(obs_array);
    cloud_global->header.frame_id = "world_link";
    mPubLaneSemanticCloud.publish(*cloud_global);
}

void StaticObstacleBuilder::ObsGlobalMerge(autoware_msgs::DetectedObjectArray obs_array)
{
    std::string stamp_string = std::to_string(obs_array.header.stamp.toSec());
    std::string obs_save_dir = mSavedir2Harddisk + "/label_4d/"; // 保存路径
    if (!boost::filesystem::exists(obs_save_dir.c_str()))
        boost::filesystem::create_directories(obs_save_dir);
    std::string obs_save_path = obs_save_dir + stamp_string + ".json";
    pcl::PointCloud<pcl::PointXYZRGBL>::Ptr cloud_edge(new pcl::PointCloud<pcl::PointXYZRGBL>);
    int label = 0;
    std::ofstream file;
    file.open(obs_save_path, std::ofstream::out);
    nlohmann::json j;
    j["obstacles"] = nlohmann::json::array();

    autoware_msgs::DetectedObjectArray obs_merge_array;
    obs_merge_array.header = obs_array.header;
    if (obs_array.objects.size() > 0)
    {
        std::vector<autoware_msgs::DetectedObject> tmp_camera_obs;
        std::vector<bool> has_merged(obs_array.objects.size(), false);
        std::vector<bool> has_merged_ed(obs_array.objects.size(), false);
        for (auto obs : obs_array.objects)
            tmp_camera_obs.push_back(obs);
        for (int i = 0; i < tmp_camera_obs.size(); ++i)
        {
            bool merge_camera_obs = false;
            pcl::PointCloud<pcl::PointXYZ>::Ptr object_single_cloud(new pcl::PointCloud<pcl::PointXYZ>);
            for (int j = 0; j < tmp_camera_obs.size() && j != i; ++j)
            {
                int label_i = tmp_camera_obs[i].label.length();
                int label_j = tmp_camera_obs[j].label.length();
                if (label_i != label_j)
                    continue;
                float inter_area = PolyIOU::calIntersectRatioCamera(tmp_camera_obs[i].convex_hull.polygon.points, tmp_camera_obs[j].convex_hull.polygon.points);
                if (inter_area >= 0.01)
                {
                    has_merged[i] = true;
                    if (has_merged[i])
                        has_merged_ed[i] = true;

                    merge_camera_obs = true;
                    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
                    pcl::fromROSMsg(tmp_camera_obs[j].pointcloud, *cloud);
                    for (auto ps : cloud->points)
                    {
                        object_single_cloud->points.push_back(ps);
                    }
                    tmp_camera_obs[j].convex_hull.polygon.points.clear();
                }
            }
            if (merge_camera_obs)
            {
                pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
                pcl::fromROSMsg(tmp_camera_obs[i].pointcloud, *cloud);
                for (auto ps : cloud->points)
                {
                    object_single_cloud->points.push_back(ps);
                }

                sensor_msgs::PointCloud2 object_cloud_msg;
                pcl::toROSMsg(*object_single_cloud, object_cloud_msg);
                object_cloud_msg.header = tmp_camera_obs[i].pointcloud.header;
                tmp_camera_obs[i].pointcloud = object_cloud_msg;

                tmp_camera_obs[i].convex_hull.polygon.points.clear();
                pcl::PointCloud<pcl::PointXYZI>::Ptr cluster_cloud(new pcl::PointCloud<pcl::PointXYZI>);
                pcl::copyPointCloud(*object_single_cloud, *cluster_cloud);
                std::vector<int> convex_indice;
                pcl::PointCloud<pcl::PointXYZ>::Ptr convex_cloud(new pcl::PointCloud<pcl::PointXYZ>);
                ClusterPolygon polygon;
                double concavity{2};
                double length_threshold{0.5};
                if (polygon.findSmallestPolygon(cluster_cloud, convex_indice))
                {
                    std::vector<std::array<double, 3>> points(cluster_cloud->size());
                    for (int j = 0; j < points.size(); j++)
                        points[j] = {cluster_cloud->points[j].x, cluster_cloud->points[j].y, static_cast<double>(j)};
                    auto concave_points = concaveman<double, 16>(points, convex_indice, concavity, length_threshold);
                    for (auto &pts : concave_points)
                    {
                        pcl::PointXYZI ps = cluster_cloud->points[static_cast<int>(pts[2])];
                        geometry_msgs::Point32 pt;
                        pt.x = ps.x;
                        pt.y = ps.y;
                        pt.z = ps.z;
                        tmp_camera_obs[i].convex_hull.polygon.points.push_back(pt);
                    }
                }
            }
        }
        for (int i = 0; i < tmp_camera_obs.size(); ++i)
        {
            if (tmp_camera_obs[i].convex_hull.polygon.points.size() < 1)
                continue;
            bool merge_camera_obs = false;
            pcl::PointCloud<pcl::PointXYZ>::Ptr object_single_cloud(new pcl::PointCloud<pcl::PointXYZ>);
            for (int j = 0; j < tmp_camera_obs.size() && j != i; ++j)
            {
                if (tmp_camera_obs[j].convex_hull.polygon.points.size() < 1)
                    continue;
                int label_i = tmp_camera_obs[i].label.length();
                int label_j = tmp_camera_obs[j].label.length();
                if (label_i != label_j)
                    continue;
                float inter_area = PolyIOU::calIntersectRatioCamera(tmp_camera_obs[i].convex_hull.polygon.points, tmp_camera_obs[j].convex_hull.polygon.points);
                if (inter_area >= 0.01)
                {
                    has_merged[i] = true;
                    if (has_merged[i])
                        has_merged_ed[i] = true;
                    merge_camera_obs = true;
                    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
                    pcl::fromROSMsg(tmp_camera_obs[j].pointcloud, *cloud);
                    for (auto ps : cloud->points)
                    {
                        object_single_cloud->points.push_back(ps);
                    }
                    tmp_camera_obs[j].convex_hull.polygon.points.clear();
                }
            }
            if (merge_camera_obs)
            {
                pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
                pcl::fromROSMsg(tmp_camera_obs[i].pointcloud, *cloud);
                for (auto ps : cloud->points)
                {
                    object_single_cloud->points.push_back(ps);
                }

                sensor_msgs::PointCloud2 object_cloud_msg;
                pcl::toROSMsg(*object_single_cloud, object_cloud_msg);
                object_cloud_msg.header = tmp_camera_obs[i].pointcloud.header;
                tmp_camera_obs[i].pointcloud = object_cloud_msg;

                tmp_camera_obs[i].convex_hull.polygon.points.clear();
                pcl::PointCloud<pcl::PointXYZI>::Ptr cluster_cloud(new pcl::PointCloud<pcl::PointXYZI>);
                pcl::copyPointCloud(*object_single_cloud, *cluster_cloud);
                std::vector<int> convex_indice;
                pcl::PointCloud<pcl::PointXYZ>::Ptr convex_cloud(new pcl::PointCloud<pcl::PointXYZ>);
                ClusterPolygon polygon;
                double concavity{2};
                double length_threshold{0.5};
                if (polygon.findSmallestPolygon(cluster_cloud, convex_indice))
                {
                    std::vector<std::array<double, 3>> points(cluster_cloud->size());
                    for (int j = 0; j < points.size(); j++)
                        points[j] = {cluster_cloud->points[j].x, cluster_cloud->points[j].y, static_cast<double>(j)};
                    auto concave_points = concaveman<double, 16>(points, convex_indice, concavity, length_threshold);
                    for (auto &pts : concave_points)
                    {
                        pcl::PointXYZI ps = cluster_cloud->points[static_cast<int>(pts[2])];
                        geometry_msgs::Point32 pt;
                        pt.x = ps.x;
                        pt.y = ps.y;
                        pt.z = ps.z;
                        tmp_camera_obs[i].convex_hull.polygon.points.push_back(pt);
                    }
                }
            }
        }

        for (int i = 0; i < tmp_camera_obs.size(); ++i)
        {
            auto obs = tmp_camera_obs[i];
            if (obs.convex_hull.polygon.points.size() > 0 && has_merged_ed[i])
            {
                nlohmann::json obs_j;
                obs_j["obj_type"] = obs.label;
                obs_j["polygons"] = nlohmann::json::array();
                obs_j["bbox"] = nlohmann::json::array();
                std::ostringstream oss;
                for (auto pts : obs.convex_hull.polygon.points)
                {
                    pcl::PointXYZRGBL surface_ps;
                    surface_ps.x = pts.x;
                    surface_ps.y = pts.y;
                    surface_ps.z = pts.z;
                    surface_ps.label = label;
                    cloud_edge->push_back(surface_ps);
                    obs_j["polygons"].push_back(pts.x);
                    obs_j["polygons"].push_back(pts.y);
                    obs_j["polygons"].push_back(pts.z);
                }
                pcl::PointCloud<pcl::PointXYZ>::Ptr object_cloud(new pcl::PointCloud<pcl::PointXYZ>);
                pcl::fromROSMsg(obs.pointcloud, *object_cloud);
                pcl::PointXYZ min_point, max_point;
                pcl::getMinMax3D(*object_cloud, min_point, max_point);
                double orientation_angle;
                float length, width, height;
                length = max_point.x - min_point.x;
                width = max_point.y - min_point.y;
                height = max_point.z - min_point.z;
                jsk_recognition_msgs::BoundingBox bounding_box;
                bounding_box.header = obs_array.header;
                bounding_box.pose.position.x = min_point.x + length / 2;
                bounding_box.pose.position.y = min_point.y + width / 2;
                bounding_box.pose.position.z = min_point.z + height / 2;
                bounding_box.dimensions.x = ((length < 0) ? -1 * length : length);
                bounding_box.dimensions.y = ((width < 0) ? -1 * width : width);
                bounding_box.dimensions.z = ((height < 0) ? -1 * height : height);

                double rz = 0;
                {
                    std::vector<cv::Point2f> points;
                    for (unsigned int i = 0; i < object_cloud->points.size(); i++)
                    {
                        cv::Point2f pt;
                        pt.x = object_cloud->points[i].x;
                        pt.y = object_cloud->points[i].y;
                        points.push_back(pt);
                    }
                    std::vector<cv::Point2f> hull;
                    cv::convexHull(points, hull);
                    for (size_t i = 0; i < hull.size() + 1; i++)
                    {
                        geometry_msgs::Point32 point;
                        point.x = hull[i % hull.size()].x;
                        point.y = hull[i % hull.size()].y;
                        point.z = max_point.z + 3;
                    }
                    cv::RotatedRect box = minAreaRect(hull);
                    rz = box.angle * 3.14 / 180;
                    bounding_box.pose.position.x = box.center.x;
                    bounding_box.pose.position.y = box.center.y;
                    bounding_box.dimensions.x = box.size.width;
                    bounding_box.dimensions.y = box.size.height;
                }
                tf::Quaternion quat = tf::createQuaternionFromRPY(0.0, 0.0, rz);
                tf::quaternionTFToMsg(quat, bounding_box.pose.orientation);

                obs.pose = bounding_box.pose;
                obs.dimensions = bounding_box.dimensions;
                obs_j["bbox"].push_back(obs.pose.position.x);
                obs_j["bbox"].push_back(obs.pose.position.y);
                obs_j["bbox"].push_back(obs.pose.position.z);
                obs_j["bbox"].push_back(obs.dimensions.x);
                obs_j["bbox"].push_back(obs.dimensions.y);
                obs_j["bbox"].push_back(obs.dimensions.z);
                obs_j["bbox"].push_back(rz);
                if (obs.dimensions.x < 0.2 && obs.dimensions.y < 0.2)
                    continue;
                if (obs.dimensions.z < 0.3 && obs.label != "PUDDLE" && obs.label != "SLIT")
                    continue;
                label++;
                j["obstacles"].push_back(obs_j);
                obs_merge_array.objects.push_back(obs);
            }
        }
        cloud_edge->header.frame_id = "world_link";
        mPubEdgeCloud.publish(*cloud_edge);
    }
    mPubObsArray.publish(obs_merge_array);
    file << j.dump(4) << std::endl;
    file.close();
    ROS_INFO("Save obs to %s", obs_save_path.c_str());
}
void StaticObstacleBuilder::PointcloudCallback(const sensor_msgs::PointCloud2::ConstPtr &msg,
                                               const nav_msgs::Odometry::ConstPtr &gps_msg,
                                               const autoware_msgs::DetectedObjectArray::ConstPtr &obj_msg)
{
    OdomCallback(gps_msg);
    Obs2dCallback(obj_msg);
    nh_.param<std::string>("/local_map_pub/save_dir2hard_disk", mSavedir2Harddisk, "/media/knight/disk2knight/depth_work_tmp");
    mHeader = msg->header;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr semi_cloud_global(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::fromROSMsg(*msg, *semi_cloud_global);
    ExtractObject(semi_cloud_global);
    mpCloudQueue->push(*semi_cloud_global);
    if (mpCloudQueue->size() > obs_fusion_frame_num_ && mpCloudQueue->size() > 0)
        mpCloudQueue->pop();
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr global_map_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    std::queue<pcl::PointCloud<pcl::PointXYZRGB>> local_cloud_map_queue(*mpCloudQueue);
    while (local_cloud_map_queue.size())
    {
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_frame(new pcl::PointCloud<pcl::PointXYZRGB>);
        *cloud_frame = local_cloud_map_queue.front();
        *global_map_cloud += *cloud_frame;
        local_cloud_map_queue.pop();
    }
    global_map_cloud->header.frame_id = "world_link";
    mPubMapCloud.publish(*global_map_cloud);
}

void StaticObstacleBuilder::Procession(void)
{
    ros::AsyncSpinner spinner(6);
    spinner.start();
    while (ros::ok())
    {
        ros::Duration(0.5).sleep();
    }
    spinner.stop();
}
int main(int argc, char **argv)
{
    ros::init(argc, argv, "StaticObstacleBuilderNode");
    ros::NodeHandle nh;
    ros::NodeHandle nh_private("~");
    StaticObstacleBuilder node(nh, nh_private);
    node.Procession();
    return 0;
}