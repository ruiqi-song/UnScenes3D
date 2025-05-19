

/*
 * @Authors: Changbin you
 * @Date: 2020-11-05 09:00:00
 * @LastEditors: Changbin you
 * @LastEditTime: 2020-11-11 20:42:23
 */
#pragma once

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <vector>
#include <float.h>
#include "concaveman.hpp"
// #include "libs.h"

class ClusterPolygon
{
public:
    typedef pcl::PointXYZI PointT;
    typedef pcl::PointCloud<PointT> PointCloudType;
    typedef std::vector<PointCloudType::Ptr> CloudPtrList;
    ClusterPolygon() {};
    bool findSmallestPolygon(const PointCloudType::Ptr inCloud, std::vector<int> &boundIndice);
    void connectPolygon(PointCloudType::Ptr bound, PointCloudType::Ptr edge);

private:
    void findStartPoint(const PointCloudType::Ptr inCloud, PointT &pStart);
    void findStartPoint(const PointCloudType::Ptr inCloud, int &startIdx);

    bool samePoint(const PointT &a, const PointT &b);
    float calLineLen(const PointT &ws, const PointT &en);
    float angleOf(const PointT &s, const PointT &d);
    float reviseAngle(float angle);
    float calculateDis(const PointT &p1, const PointT &p2);
};
