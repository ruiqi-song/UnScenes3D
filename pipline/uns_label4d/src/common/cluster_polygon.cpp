

/***
 * @brief:
 * @Version: v0.0.1
 * @Author: knightdby  && knightdby@163.com
 * @Date: 2025-05-19 14:08:20
 * @Description:
 * @LastEditors: knightdby
 * @LastEditTime: 2025-05-19 14:08:20
 * @FilePath: /UnScenes3D/pipline/uns_label4d/src/common/cluster_polygon.cpp
 * @Copyright 2025 by Inc, All Rights Reserved.
 * @2025-05-19 14:08:20
 */

#include "cluster_polygon.h"

bool ClusterPolygon::findSmallestPolygon(const PointCloudType::Ptr inCloud, std::vector<int> &boundIndice)
{
    assert(inCloud != NULL);
    if (inCloud->size() == 0)
        return false;
    int cornerIdx;
    findStartPoint(inCloud, cornerIdx);
    PointT corner = inCloud->points[cornerIdx];
    double minAngleDif, oldAngle = 2 * M_PI;
    int iter = 0;
    do
    {
        boundIndice.push_back(cornerIdx);
        minAngleDif = 2 * M_PI;

        PointT nextPoint = corner;
        int nextPointIdx = cornerIdx;
        double nextAngle = oldAngle;
        for (int i = 0; i < inCloud->points.size(); i++)
        {
            PointT p = inCloud->points[i];
            if (p.intensity == 1)
            { // 已被加入边界链表的点
                continue;
            }
            if (samePoint(p, corner))
            { // 重合点
                continue;
            }
            float currAngle = angleOf(corner, p);               /* 当前向量与x轴正方向的夹角 */
            float angleDif = reviseAngle(oldAngle - currAngle); /* 两条向量之间的夹角（顺时针旋转的夹角） */

            if (angleDif < minAngleDif)
            {
                minAngleDif = angleDif;
                nextPoint = p;
                nextPointIdx = i;
                nextAngle = currAngle;
            }
        }

        iter++;
        if (iter > inCloud->size())
            return false;

        oldAngle = nextAngle;
        corner = nextPoint;
        cornerIdx = nextPointIdx;
        inCloud->points[cornerIdx].intensity = 1;
    } while (!samePoint(corner, inCloud->points[boundIndice[0]]));

    return true;
}

void ClusterPolygon::connectPolygon(PointCloudType::Ptr bound, PointCloudType::Ptr edge)
{
    //    *edge = *bound;
    //    bound->points.push_back(bound->points[0]);
    //    float resolution = 0.2f;
    //    for(int i = 0; i < bound->points.size() - 1; i++){
    //        PointT p1 = bound->points[i];
    //        PointT p2 = bound->points[i + 1];
    //
    //        float dis = sqrt(pow(p1.x - p2.x, 2) + pow(p1.y - p2.y, 2) + pow(p1.z - p2.z, 2));
    //        int insertNum = ceil(dis / resolution);
    //
    //        for(int m = 1; m <= insertNum; m++){
    //            PointT addPt;
    //            addPt.x = (m * p1.x + (insertNum + 1 - m) * p2.x) / (insertNum + 1);
    //            addPt.y = (m * p1.y + (insertNum + 1 - m) * p2.y) / (insertNum + 1);
    //            addPt.z = (m * p1.z + (insertNum + 1 - m) * p2.z) / (insertNum + 1);
    //            edge->points.push_back(addPt);
    //        }
    //    }

    bound->points.push_back(bound->points[0]);
    float resolution = 0.2f;
    int iten = 0;
    for (int i = 0; i < static_cast<int>(bound->points.size()) - 1; i++)
    {
        PointT p1 = bound->points[i];
        PointT p2 = bound->points[i + 1];
        if (i == 0)
        {
            iten++;
            p1.intensity = iten;
            edge->points.push_back(p1);
        }
        float dis = sqrt(pow(p1.x - p2.x, 2) + pow(p1.y - p2.y, 2) + pow(p1.z - p2.z, 2));
        int insertNum = ceil(dis / resolution);

        for (int m = insertNum; m >= 1; m--)
        {
            iten++;
            PointT addPt;
            addPt.x = (m * p1.x + (insertNum + 1 - m) * p2.x) / (insertNum + 1);
            addPt.y = (m * p1.y + (insertNum + 1 - m) * p2.y) / (insertNum + 1);
            addPt.z = (m * p1.z + (insertNum + 1 - m) * p2.z) / (insertNum + 1);
            addPt.intensity = iten;
            edge->points.push_back(addPt);
        }

        if (i != static_cast<int>(bound->points.size()) - 2)
        {
            iten++;
            p2.intensity = iten;
            edge->points.push_back(p2);
        }
    }
}

bool ClusterPolygon::samePoint(const PointT &a, const PointT &b)
{
    if (fabs(a.x - b.x) < 0.00001 && fabs(a.y - b.y) < 0.00001)
        return true;
    return false;
}

float ClusterPolygon::calLineLen(const PointT &ws, const PointT &en)
{
    if (samePoint(ws, en))
    {
        return .0;
    }

    float a = fabs(ws.x - en.x); // 直角三角形的直边a
    float b = fabs(ws.y - en.y); // 直角三角形的直边b

    float minEdge = std::min(a, b); // 短直边
    float maxEdge = std::max(a, b); // 长直边

    float inner = minEdge / maxEdge;
    return sqrt(inner * inner + 1.0) * maxEdge;
}

float ClusterPolygon::angleOf(const PointT &s, const PointT &d)
{
    float dist = calLineLen(s, d);

    if (dist <= 0)
    {
        return .0;
    }

    float x = d.x - s.x; // 直角三角形的直边a
    float y = d.y - s.y; // 直角三角形的直边b

    if (y >= 0.)
    { /* 1 2 象限 */
        return acos(x / dist);
    }
    else
    { /* 3 4 象限 */
        return acos(-x / dist) + M_PI;
    }
}

float ClusterPolygon::reviseAngle(float angle)
{
    while (angle < 0.)
    {
        angle += 2 * M_PI;
    }
    while (angle >= 2 * M_PI)
    {
        angle -= 2 * M_PI;
    }
    return angle;
}

void ClusterPolygon::findStartPoint(const PointCloudType::Ptr inCloud, PointT &pStart)
{
    pStart = inCloud->points[0];
    for (int i = 1; i < inCloud->points.size(); i++)
    {
        PointT pNext = inCloud->points[i];
        if (pNext.y > pStart.y || (pNext.y == pStart.y && pNext.x < pStart.x))
            pStart = pNext;
    }
}

void ClusterPolygon::findStartPoint(const PointCloudType::Ptr inCloud, int &startIdx)
{
    PointT pStart = inCloud->points[0];
    startIdx = 0;
    for (int i = 1; i < inCloud->points.size(); i++)
    {
        PointT pNext = inCloud->points[i];
        if (pNext.y > pStart.y || (pNext.y == pStart.y && pNext.x < pStart.x))
        {
            pStart = pNext;
            startIdx = i;
        }
    }
}

float ClusterPolygon::calculateDis(const PointT &p1, const PointT &p2)
{
    return sqrt(pow(p1.x - p2.x, 2) + pow(p1.y - p2.y, 2));
}
