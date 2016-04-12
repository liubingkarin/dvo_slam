#ifndef COMMON_H_
#define COMMON_H_

// pcl
//#include <pcl/ros/conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/time.h>
//#include <pcl_ros/transforms.h>

//using namespace Eigen;

typedef pcl::PointXYZRGB PointT;
typedef pcl::PointCloud<PointT> Cloud;
typedef Cloud::Ptr CloudPtr;
typedef Cloud::ConstPtr CloudConstPtr;

typedef pcl::PointXYZRGBNormal PointNT;
typedef pcl::PointCloud<PointNT> CloudN;
typedef CloudN::Ptr CloudNPtr;


typedef Eigen::Matrix<float,6,1> Vector6f;



inline float angDiff(float ang1, float ang2)
{
    float diff = ang1-ang2;
    while(diff > M_PI) {diff -= 2.f*M_PI;}
    while(diff < -M_PI) {diff += 2.f*M_PI;}
    return diff;
}

inline float colorsimRGB(PointT a, PointT b)
{
    float r1,g1,b1,r2,g2,b2;
    r1 = float(a.r); r2= float(b.r);
    g1 = float(a.g); g2= float(b.g);
    b1 = float(a.b); b2= float(b.b);
//    float maxdist= sqrt(255.f*255.f *3);
//    return 1.f-sqrt(pow(r1-r2, 2) + pow(b1-b2,2) + pow(g1-g2,2))/maxdist;
    return (1.f-sqrt(pow(r1-r2,2) + pow(b1-b2,2) + pow(g1-g2,2))/441.7f);
}

inline float colorsimGray(PointT a, PointT b)
{
    float r1,g1,b1,r2,g2,b2;
    r1 = float(a.r); r2= float(b.r);
    g1 = float(a.g); g2= float(b.g);
    b1 = float(a.b); b2= float(b.b);
    return 1.f-fabs(0.299f*r1+0.587f*g1+0.114f*b1-0.299f*r2-0.587f*g2-0.114f*b2)/255.f;
}

inline float getresidual(PointT a, PointT b)
{
    float r1,g1,b1,r2,g2,b2;
    r1 = float(a.r); r2= float(b.r);
    g1 = float(a.g); g2= float(b.g);
    b1 = float(a.b); b2= float(b.b);
    return (0.299f*r1+0.587f*g1+0.114f*b1-0.299f*r2-0.587f*g2-0.114f*b2);
}



#endif // COMMON_H
