/***
 * @brief:
 * @Author: dingbaiyong && baiyong.ding@waytous.com
 * @Date: 2024-09-18 09:28:35
 * @FilePath: /god_depth/include/common/octree_connection.hpp
 * @Description:
 * @LastEditTime: 2024-09-18 09:28:54
 * @LastEditors: dingbaiyong
 * @Copyright (c) 2024 by Inc, All Rights Reserved.
 */
#ifndef OCTREE_CONNECTION
#define OCTREE_CONNECTION

#define PCL_NO_PRECOMPILE

#include <pcl/point_types.h>
#include <pcl/conversions.h>
#include <pcl/common/centroid.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/plane_clipper3D.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/conditional_euclidean_clustering.h>
#include <pcl/features/moment_of_inertia_estimation.h>
#include <Eigen/Dense>
#include <vector>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/io/pcd_io.h> //文件输入输出
                           /*
                            *@brief  输入需要聚类的点云，参数；输出聚类完成带标签的点云
                            */
class OTC
{

private:
    /*
     * @brief  计算点云至原点的距离
     * @param  points    输入三维点
     * @return double    返回距离长度
     */
    double get_points2dis(pcl::PointXYZ points)
    {
        double x = points.x;
        double y = points.y;
        double z = points.z;
        return pow(pow(x, 2) + pow(y, 2) + pow(z, 2), 0.5);
    };

    float octree_leaf_size = 0.3f; ///< 八叉树深度参数
    int max_points_size = 20000;   ///< 聚类最大点云数
    int min_points_size = 2;       ///< 聚类最小点云数
    int sourch_k_num = 10;         ///< 聚类K近邻搜索数
    double dis_th = 0.1;           ///< 聚类放大比例系数

public:
    /*
     * @brief  八叉树结构去噪算法
     * @param  cloud         输入需要去噪点云
     * @param  filter_cloud  输出去噪后的点云
     * @return bool          返回是否成功去噪
     */
    bool octree_denoise(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud,
                        pcl::PointCloud<pcl::PointXYZ> &filter_cloud);

    /*
     * @brief  设定八叉树最小深度值
     * @param  size          输入定义深度值
     */
    void set_octree_leafsize(const float size)
    {
        octree_leaf_size = size;
    };

    /*
     * @brief  设定聚类最小值
     * @param  min_size       输入定义聚类最小值
     */
    void set_min_points_size(const int min_size)
    {
        min_points_size = min_size;
    };

    /*
     * @brief  设定聚类比例系数
     * @param  distance_th    输入定义聚类比例系数
     */
    void set_distance_th(const double distance_th)
    {
        dis_th = distance_th;
    };

    /*
     * @brief  欧几里得聚类算法(采用KNN搜索)
     * @param  cloud         输入需要聚类的点云
     * @param  tree          输入搜索方式，默认为KDtree结构
     * @param  tolerance     输入八叉树搜索间隔
     * @param  clusters      输出聚类后的类别集合
     * @param  min_pts_per_cluster    输入聚类点云的最小值
     * @param  max_pts_per_cluster    输入聚类点云的最大值
     */
    void euclidean_clusters(const pcl::PointCloud<pcl::PointXYZ> cloud,
                            const typename pcl::search::Search<pcl::PointXYZ>::Ptr &tree,
                            double tolerance, std::vector<pcl::PointIndices> &clusters,
                            unsigned int min_pts_per_cluster,
                            unsigned int max_pts_per_cluster);

    /*
     * @brief  八叉树连通域聚类算法
     * @param  input_cloud           输入需要聚类的点云
     * @param  octree_connect_cloud  输出聚类后的点云集合
     */
    void octree_connection(const pcl::PointCloud<pcl::PointXYZ>::Ptr &input_cloud,
                           pcl::PointCloud<pcl::PointXYZ> &filter_cloud);
    OTC() {};

    ~OTC() {};
};
void OTC::euclidean_clusters(const pcl::PointCloud<pcl::PointXYZ> cloud,
                             const typename pcl::search::Search<pcl::PointXYZ>::Ptr &tree,
                             double tolerance,
                             std::vector<pcl::PointIndices> &clusters,
                             unsigned int min_pts_per_cluster,
                             unsigned int max_pts_per_cluster)
{
    if (tree->getInputCloud()->points.size() != cloud.points.size())
    {
        PCL_ERROR("[pcl::extracteuclidean_clusters] Tree built for a different point cloud dataset (%lu) than the input cloud (%lu)!\n", tree->getInputCloud()->points.size(), cloud.points.size());
        return;
    }
    int nn_start_idx = tree->getSortedResults() ? 1 : 0;
    std::vector<bool> processed(cloud.points.size(), false);
    std::vector<int> nn_indices;
    std::vector<float> nn_distances;

    // 遍历点云中的每一个点
    for (int i = 0; i < static_cast<int>(cloud.points.size()); ++i)
    {
        if (processed[i]) ///< 如果该点已经处理则跳过
            continue;
        std::vector<int> seed_queue; ///< 定义一个种子队列
        int sq_idx = 0;
        seed_queue.push_back(i); ///< 加入一个种子
        processed[i] = true;
        while (sq_idx < static_cast<int>(seed_queue.size()))
        {
            /*采用KNN搜索方式进行相邻点判定*/
            if (!tree->nearestKSearch(seed_queue[sq_idx], sourch_k_num, nn_indices, nn_distances))
            {
                sq_idx++;
                continue; ///< 没找到近邻点就继续
            }
            double dis = get_points2dis(cloud.points[seed_queue[sq_idx]]);
            for (size_t j = nn_start_idx; j < nn_indices.size(); ++j)
            {
                if (nn_indices[j] == -1 || processed[nn_indices[j]])
                    continue; ///< 种子点的近邻点中如果已经处理就跳出此次循环继续
                if (nn_distances[j] > std::max(dis_th * tolerance * dis, tolerance))
                    continue;
                seed_queue.push_back(nn_indices[j]); ///< 将此种子点的临近点作为新的种子点。入队操作
                processed[nn_indices[j]] = true;     ///< 该点已经处理，打标签
            }
            sq_idx++;
        }

        /*最大点数和最小点数的类过滤*/
        if (seed_queue.size() >= min_pts_per_cluster && seed_queue.size() <= max_pts_per_cluster)
        {
            pcl::PointIndices r;
            r.indices.resize(seed_queue.size());
            for (size_t j = 0; j < seed_queue.size(); ++j)
                r.indices[j] = seed_queue[j];

            std::sort(r.indices.begin(), r.indices.end());
            r.indices.erase(std::unique(r.indices.begin(), r.indices.end()), r.indices.end());

            r.header = cloud.header;
            clusters.push_back(r);
        }
    }
}
void OTC::octree_connection(const pcl::PointCloud<pcl::PointXYZ>::Ptr &input_cloud,
                            pcl::PointCloud<pcl::PointXYZ> &filter_cloud)
{
    std::vector<pcl::PointCloud<pcl::PointXYZ>> octree_connect_cloud;
    std::vector<pcl::PointXYZ, Eigen::aligned_allocator<pcl::PointXYZ>> voxel_center_list_arg;
    voxel_center_list_arg.clear();
    octree_connect_cloud.clear();
    pcl::octree::OctreePointCloudSearch<pcl::PointXYZ> octree(octree_leaf_size);
    octree.setInputCloud(input_cloud);
    octree.addPointsFromInputCloud();
    octree.getOccupiedVoxelCenters(voxel_center_list_arg);
    //  等同于 occupiedVoxelCenters
    pcl::PointCloud<pcl::PointXYZ> v_cloud, euc_cloud;
    v_cloud.resize(voxel_center_list_arg.size());
    for (size_t i = 0; i < voxel_center_list_arg.size(); i++)
    {
        v_cloud[i].x = voxel_center_list_arg[i].x;
        v_cloud[i].y = voxel_center_list_arg[i].y;
        v_cloud[i].z = voxel_center_list_arg[i].z;
    }

    // 欧几里得聚类
    float dis_th = pow(pow((octree_leaf_size * 2), 2) + pow((octree_leaf_size * 2), 2), 0.5) + 0.001; ///< 计算聚类深度阈值
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
    std::vector<pcl::PointIndices> ece_inlier;
    tree->setInputCloud(v_cloud.makeShared());
    euclidean_clusters(v_cloud, tree, dis_th, ece_inlier, 1, max_points_size); ///< 聚类
    int num = 0;
    for (int i = 0; i < ece_inlier.size(); i++)
    {
        /*聚类完成后，需要重新找到八叉树内部所有点(体素搜索)*/
        std::vector<int> ece_inlier_ext = ece_inlier[i].indices; ///< 输入所聚类到的体素中心点
        pcl::PointCloud<pcl::PointXYZ> voxel_cloud, cloud_copy;
        pcl::copyPointCloud(v_cloud, ece_inlier_ext, cloud_copy); ///< 按照索引提取点云数据
        for (int j = 0; j < cloud_copy.points.size(); j++)
        {

            std::vector<int> pointIdxVec; ///< 保存体素近邻搜索的结果向量
            if (octree.voxelSearch(cloud_copy.points[j], pointIdxVec))
            {
                for (size_t k = 0; k < pointIdxVec.size(); ++k)
                {
                    voxel_cloud.push_back(input_cloud->points[pointIdxVec[k]]);
                }
            }
        }
        if (voxel_cloud.points.size() > min_points_size)
        {
            num++;
            octree_connect_cloud.push_back(voxel_cloud);
        }
    }
    int max_cloud_idx = 0;
    int max_cloud_num = 0;

    for (size_t k = 0; k < octree_connect_cloud.size(); ++k)
    {
        if (octree_connect_cloud[k].size() > max_cloud_num)
        {
            max_cloud_num = octree_connect_cloud[k].size();
            max_cloud_idx = k;
        }
    }
    filter_cloud = octree_connect_cloud[max_cloud_idx];
    // std::cout << "otc filter: " << input_cloud->size() << ", " << filter_cloud.size() << ", " << octree_connect_cloud.size() << std::endl;
}

#endif
