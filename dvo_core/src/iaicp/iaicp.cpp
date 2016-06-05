#include "iaicp/iaicp.h"
#include <Eigen/Geometry>
#include <fstream>
#include <cmath>

using namespace pcl;
using namespace pcl::registration;
using namespace Eigen;
using namespace std;
using namespace cv;

Iaicp::Iaicp()
{
    //ros::param::get("/fx", fx);
    //ros::param::get("/fy", fy);
    //ros::param::get("/cx", cx);
    //ros::param::get("/cy", cy);
    //ros::param::get("/height", height);
    //ros::param::get("/width", width);
    fx = 517.5;
    fy = 516.5;
    cx = 318.6;
    cy = 255.3;
//    fx = 525.0 ;// # focal length x
//    fy = 525.0  ;//# focal length y
//    cx = 319.5  ;//# optical center x
//    cy = 239.5  ;//# optical center y
    m_trans = Affine3f::Identity();
    m_predict = Affine3f::Identity();
    width = 640;
    height = 480;

    interatePerLevel = 15;
    numOfFeaturePairs = 100;
    searchRangePixel = 12;
}

Iaicp::~Iaicp(){}

CloudPtr Iaicp::Mat2Cloud(Mat &imR, Mat &imD)
{
    CloudPtr cloud;
    cloud.reset(new Cloud);

    cloud->points.resize(width*height);

    for(int c=0; c<width; c++){
        for(int r=0; r<height; r++){
            PointT pt;
            if ( isnan(imD.at<float>(r,c)) ) {
                pt.z= nan("");
                cloud->points.at(r*width+c) = pt;
                continue;
            }
            pt.z = imD.at<float>(r,c);
            pt.x = (float(c) - cx) * pt.z / fx;
            pt.y = (float(r) - cy) * pt.z / fy;

            Vec3b color = imR.at<Vec3b>(r,c);
            pt.r = (int)color.val[0];
            pt.g = (int)color.val[1];
            pt.b = (int)color.val[2];
            cloud->points.at(r*width+c) = pt;
        }
    }
    return cloud;
}
void Iaicp::setupSource(CloudPtr &source)
{
    m_src.reset(new Cloud);
    m_src = source;
    intMedian=0.f;  intMad=45.f;
}

void Iaicp::setupTarget(CloudPtr &target)
{
    m_tgt.reset(new Cloud);
    m_tgt = target;
    geoMedian =0.f; geoMad=0.02f;
}

void Iaicp::setupPredict(Affine3f pred)
{
    m_predict = pred;
    m_trans = pred;
}

void Iaicp::setupPredict(Affine3d pred)
{
    m_predict = pred.cast<float>();
    m_trans = pred.cast<float>();
}


void Iaicp::sampleSource()
{
    //cout<<m_src->points.size()<<"  "<<m_tgt->points.size()<<endl;
    m_salientSrc.reset(new Cloud());
    int cnt=0;
    int begin_r = 2 + rand()%2;
    int begin_c = 2 + rand()%2;
    //cout<<width<<"  "<<height<<endl;
    for(size_t i= begin_c; i<width-begin_c-4; i+=4){
        for(size_t j=begin_r; j<height-begin_r-4; j+=4){
            PointT pt = m_src->points[j*width+i];
            if (pt.z!=pt.z || pt.z>8.f) {continue;}  //continue if no depth value available

            //warp to target image
            PointT ptwarp=pt;
            ptwarp = pcl::transformPoint(ptwarp, m_predict);
            if(isnan(ptwarp.x) || isnan(ptwarp.y) || isnan(ptwarp.z)){continue;}

            int xpos = round(fx/ptwarp.z * ptwarp.x + cx);
            int ypos = round(fy/ptwarp.z * ptwarp.y + cy);
            if (xpos>=width-3 || ypos>=height-3 || xpos<3 || ypos<3) {continue;} //continue if out of image border


            //check whether backgfloor point
            float z_ = m_src->points[j*width+i].z;
            float diff1=z_ - m_src->points[j*width+i+2].z;
            float diff2=z_ - m_src->points[j*width+i-2].z;
            float diff3=z_ - m_src->points[(j-2)*width+i].z;
            float diff4=z_ - m_src->points[(j+2)*width+i].z;
            if ( diff1!=diff1 || diff2!=diff2 || diff3!=diff3 || diff4!=diff4){
                continue;
            }
            float thres= 0.021*z_;
            if ( diff1>thres || diff2>thres || diff3>thres || diff4>thres ){
                continue;
            }

            //            //image gradient
            float sim1=colorsimGray(m_src->points[j*width+i-4], m_src->points[j*width+i+4] );
            float sim2=colorsimGray(m_src->points[(j-4)*width+i], m_src->points[(j+4)*width+i] );
            if((sim1==sim1 && sim1<=0.85f)||(sim2==sim2 && sim2 <=0.85f)){
                m_salientSrc->points.push_back(m_src->points[j*width+i]);
                cnt++;
                continue;
            }

            //intensity residual
            float residual= fabs(getresidual(m_tgt->points[ypos*width+xpos], pt));
            if (fabs(residual)>100.f){
                m_salientSrc->points.push_back(m_src->points[j*width+i]);
                cnt++;
                continue;
            }

            //depth gradient
            if ( fabs(diff1-diff2) >0.03f*z_ || fabs(diff3-diff4) >0.03f*z_){
                m_salientSrc->points.push_back(m_src->points[j*width+i]);
                cnt++;
                continue;
            }

            //            //large depth change
            //            float zdiff = pt.z - m_tgt->points[ypos*width+xpos].z;
            //            if (fabs(zdiff)>0.09f*z_){
            //                m_salientSrc->points.push_back(m_src->points[j*width+i]);
            //                cnt++;
            //                continue;
            //            }
        }
    }

    if(cnt<200){
        for(size_t i=0; i<1000; i++){
            m_salientSrc->points.push_back(m_src->points[rand()%m_src->points.size()]);
        }
    }
    vector<int> indices;
    pcl::removeNaNFromPointCloud(*m_salientSrc, *m_salientSrc, indices);
    //cout<<"sampled "<< cnt<<" salient points"<<endl;
}
int Iaicp::getSearchRangePixel() const
{
    return searchRangePixel;
}

void Iaicp::setSearchRangePixel(int value)
{
    searchRangePixel = value;
}

int Iaicp::getNumOfFeaturePairs() const
{
    return numOfFeaturePairs;
}

void Iaicp::setNumOfFeaturePairs(int value)
{
    numOfFeaturePairs = value;
}

int Iaicp::getInteratePerLevel() const
{
    return interatePerLevel;
}

void Iaicp::setInteratePerLevel(int value)
{
    interatePerLevel = value;
}


void Iaicp::run()
{
    sampleSource();
    int iterPerLevel = interatePerLevel;
    
    int offset=6, maxDist=0.12f;
    iterateLevel(maxDist, offset, iterPerLevel);
    offset=3; maxDist=0.06f;
    iterateLevel(maxDist, offset, iterPerLevel);
    offset=1; maxDist=0.02f;
    iterateLevel(maxDist, offset, iterPerLevel);
}

void Iaicp::iterateLevel(float maxDist, int offset, int maxiter) //performs one iteration of the IaICP method
{
    for (size_t iteration=0; iteration < maxiter; iteration++){
        tgt_.reset(new Cloud());
        src_.reset(new Cloud());
        std::vector<float> geoResiduals;
        std::vector<float> intResiduals;

        int counter=0;   //counter for how many number of correspondences have been already used
        for(size_t i = 0; i < m_salientSrc->points.size(); i++){
            if (counter >= numOfFeaturePairs) break;    //only use  limited number of pairs of correspondences.
            int thisIndex =  rand()%m_salientSrc->points.size();  //randomly select one salient point
            PointT temp = m_salientSrc->points[thisIndex];   //selected source ponint
            PointT pt = transformPoint(temp, m_trans);   //warped source point
            PointT tgtpt;                              //for the selected correponding point from the target cloud.
            int xpos = int(round(fx/pt.z * pt.x + cx)); //warped image coordinate x
            int ypos = int(round(fy/pt.z * pt.y + cy)); //warped image coordinate y

            if ((xpos >= width) || (ypos >= height)|| xpos<0 || ypos<0) { continue;}
            float maxWeight = 1e-10;
            int searchRange = searchRangePixel;//3;
            float intResidual, geoResidual;

            for(int xx = -searchRange; xx < searchRange+1; xx++){
                for(int yy = -searchRange; yy < searchRange+1; yy++){
                    float gridDist = sqrt(pow(float(xx),2) + pow(float(yy),2));
                    if ( gridDist > (float)searchRange ){continue;}  //get a circle shaped search area

                    int xpos_ = xpos + xx*offset;  //searched target point's image coordinate
                    int ypos_ = ypos + yy*offset;// + floor(rand()%offset - 0.5*offset)

                    if (xpos_>=(width-2) || ypos_>=(height-2) || xpos_<2 || ypos_<2) { continue;}

                    PointT pt2 = m_tgt->points[ypos_*width+xpos_];
                    float dist = (pt.getVector3fMap() - pt2.getVector3fMap()).norm();  //geo. distance
                    if(dist==dist){           //check for NAN
                        //                        if (dist>maxDist) {continue;}
                        float residual = getresidual(pt2, pt);
                        if(residual == residual){  //check for NAN
                            float geoWeight = 1e2f*(6.f/(5.f+ pow((dist)/(geoMad), 2)));
                            float colWeight = 1e2f*(6.f/(5.f+ pow((residual-intMedian)/intMad, 2)));
                            float thisweight = geoWeight * colWeight;
                            if(thisweight == thisweight && thisweight>maxWeight){
                                tgtpt=pt2;
                                maxWeight=thisweight;
                                intResidual= residual; geoResidual = dist;
                            }
                        }
                    }
                }
            }

            if(maxWeight>0 ){
                if ((m_salientSrc->points[thisIndex].getVector3fMap()-tgtpt.getVector3fMap()).norm()<1000.f){
                    src_->points.push_back(pt);
                    tgt_->points.push_back(tgtpt);

                    intResidual=getresidual(tgtpt, pt);
                    geoResidual = (pt.getVector3fMap()-tgtpt.getVector3fMap()).norm();
                    intResiduals.push_back(intResidual);
                    geoResiduals.push_back(geoResidual);
                    counter++;
                }
            }
        }

        //Estimate median and deviation for both intensity and geometry residuals
        vector<float> temp = geoResiduals;
        sort(temp.begin(), temp.end());
        geoMedian = temp[temp.size()-temp.size()/2];
        for(size_t i=0; i<temp.size(); i++){
            temp[i] = fabs(temp[i]-geoMedian);
        }
        sort(temp.begin(), temp.end());
        geoMad = 1.f*1.4826 * temp[temp.size()/2] + 1e-11;
        for(size_t i=0; i<geoResiduals.size(); i++){
            geoResiduals[i] =  (6.f/(5.f+ pow((geoResiduals[i])/geoMad, 2)));
        }
        temp.clear();
        temp = intResiduals;
        sort(temp.begin(), temp.end());
        intMedian = temp[temp.size()-temp.size()/2];
        for(size_t i=0; i<temp.size(); i++){
            temp[i] = fabs(temp[i]-intMedian);
        }
        sort(temp.begin(), temp.end());
        intMad = 1.f*1.4826 * temp[temp.size()/2] + 1e-11;
        for(size_t i=0; i<intResiduals.size(); i++){
            intResiduals[i] = (6.f/(5.f+ pow((intResiduals[i]-intMedian)/intMad, 2)));
        }

        pcl::TransformationFromCorrespondences transFromCorr;
        for (size_t i =0;i<src_->points.size();i++)
        {
            Eigen::Vector3f from(src_->points.at(i).x, src_->points.at(i).y, src_->points.at(i).z);
            Eigen::Vector3f to(tgt_->points.at(i).x, tgt_->points.at(i).y, tgt_->points.at(i).z);
            float sensorRel = 1.f;///(0.0012+0.0019*pow(src_->points.at(i).z-0.4, 2));
            transFromCorr.add(from, to, geoResiduals[i] * intResiduals[i]*sensorRel);

        }
        Affine3f increTrans= transFromCorr.getTransformation();
        m_trans = toEigen(toVector(increTrans *m_trans) ) ;

    }
}



void Iaicp::checkAngles(Vector6f &vec)
{
    for(size_t i=3; i<6; i++){
        while (vec(i)>M_PI)  {vec(i) -= 2*M_PI;}
        while (vec(i)<-M_PI) {vec(i) += 2*M_PI;}
    }
}

Affine3f Iaicp::toEigen(Vector6f pose)
{
    return pcl::getTransformation(pose(0),pose(1),pose(2),pose(3),pose(4),pose(5));
}

Vector6f Iaicp::toVector(Affine3f pose)
{
    Vector6f temp;
    pcl::getTranslationAndEulerAngles(pose, temp(0),temp(1),temp(2),temp(3),temp(4),temp(5));
    checkAngles(temp);
    return temp;
}

Mat Iaicp::cloudToImage(const CloudPtr &cloud, Affine3f transform)
{
    std::vector<double> proj_depth(width*height);//warped image

    for(int i = 0; i < width*height; i++)
    {
        proj_depth[i] = 1e10;
    }
    for(int c = 0; c < width; c++)
    {
        for(int r = 0; r < height; r++)
        {
            PointT point = cloud->points[r*width+c];

            //warp
            point = transformPoint(point, transform);

            int c_t = round(point.x*fx/point.z + cx);//!!!point.z could be nan
            int r_t = round(point.y*fy/point.z + cy);

            if (c_t >=0 && c_t < width && r_t >=0 && r_t < height)
            {
                if(point.z < proj_depth[r_t*width + c_t])
                {
                    proj_depth[r_t*width + c_t] = point.z;

                }
            }
        }
    }
    Mat mat(height, width, CV_32F);
    for(size_t c= 0; c < width; c++)
    {
        for(size_t r = 0; r < height; r++)
        {

            if(proj_depth[r*width+c] > 10000){
                mat.row(r).col(c) = 10000;
            }
            else
            {
                mat.row(r).col(c) = std::max(std::min(proj_depth[r*width+c]/4.0, 1.0), 0.0);
            }
        }
    }

    return mat;
}

void Iaicp::writeResidualImgToFile(Affine3f transform, string fileName) const
{
    std::vector<double> proj_depth(width*height);//warped image

    for(int i = 0; i < width*height; i++)
    {
        proj_depth[i] = 1e10;
    }
    for(size_t c = 0; c < width; c++)
    {
        for(size_t r = 0; r < height; r++)
        {
            PointT point = m_src->points[r*width+c];

            //warp
            point = transformPoint(point, transform);

            int c_t = round(point.x*fx/point.z + cx);
            int r_t = round(point.y*fy/point.z + cy);

            if (c_t >=0 && c_t < width && r_t >=0 && r_t < height)
            {
                if(point.z < proj_depth[r_t*width + c_t])
                {
                    proj_depth[r_t*width + c_t] = point.z;
                }
            }
        }
    }

    ofstream myfile;
    myfile.open (fileName.c_str());
    for(int i = 0; i < width*height; i++)
    {
        PointT p = m_tgt->points[i];
        if(proj_depth[i] < 1e5 && p.z == p.z)
        {

            myfile<<proj_depth[i]-p.z<<endl;
        }
    }
    myfile.close();
}

void Iaicp::llhAndInfomatrix(Affine3f transform, double &llh, dvo::core::Matrix6d &Information)
{
    std::vector<double> proj_depth(width*height);//warped image

    //warp
    for(int i = 0; i < width*height; i++)
    {
        proj_depth[i] = 1e10;
    }
    for(size_t c = 0; c < width; c++)
    {
        for(size_t r = 0; r < height; r++)
        {
            PointT point = m_src->points[r*width+c];

            //warp
            point = transformPoint(point, transform);
            if(isnan(point.z)){continue;}

            int c_t = round(point.x*fx/point.z + cx);
            int r_t = round(point.y*fy/point.z + cy);

            if (c_t >=0 && c_t < width && r_t >=0 && r_t < height)
            {
                if(point.z < proj_depth[r_t*width + c_t])
                {
                    proj_depth[r_t*width + c_t] = point.z;
                }
            }
        }
    }



    //calculate mean absolute error (loglikelihood)
    double errorSum = 0;
    double negErrSum = 0;
    int count = 0;
    double goodSum = 1.0;
    for(int i = 0; i < width*height; i++)
    {
        PointT p = m_tgt->points[i];
        if(proj_depth[i] < 1e5 &&  p.z == p.z)
        {
            errorSum += abs(proj_depth[i] - p.z);
            negErrSum += std::max((0.02 - abs(proj_depth[i]-p.z)), 0.001);
            if(abs(proj_depth[i]-p.z) < 0.005){
                goodSum += 1.0;
            }
            if(abs(proj_depth[i]-p.z) > 0.02){
                goodSum -= 1.0;
            }
            count++;
        }
        //        else if(p.z != p.z)
        //        {
        //            cout<<"min(nan,1) "<<std::min(p.z,1.0f)<<endl;
        //            cout<<"max(nan,0) "<<std::max(p.z,0.0f)<<endl;
        //            cout<<"max(std::min(nan,1.0f),0) "<<std::max(std::min(p.z,1.0f),0.0f)<<endl;
        //        }
    }

    double mean = errorSum/count;
    llh = -std::max(negErrSum, 0.01)/count*1000;
    //llh = -std::max(goodSum, 1.0)/count;

    //calculate variance --> informationmatrix
    double varianceSum = 0;
    count = 0;
    for(int i = 0; i < width*height; i++)
    {
        PointT p = m_tgt->points[i];
        if(proj_depth[i] < 1e5 &&  p.z == p.z)
        {
            varianceSum += (abs(proj_depth[i]-p.z) - mean) * (abs(proj_depth[i]-p.z) - mean);
            count++;
        }
    }
    double variance = varianceSum / count;
    variance = 0.008;

    Information = dvo::core::Matrix6d::Identity() * variance * variance;
    //cout << "variance  " << variance;

}

bool Iaicp::match(dvo::core::PointSelection &ref, dvo::core::RgbdImagePyramid &cur, dvo::DenseTracker::Result &result)
{
    match(ref.getRgbdImagePyramid(), cur, result);
}

bool Iaicp::match(dvo::core::RgbdImagePyramid &ref, dvo::core::RgbdImagePyramid &cur, dvo::DenseTracker::Result &result)
{
    double time_start = pcl::getTime();
    Eigen::Affine3f result_key;
    CloudPtr tmp_s, tmp_t;
    double time_core_start = pcl::getTime();

    tmp_s = Mat2Cloud(cur.ori_rgb, cur.ori_depth);

    tmp_t = Mat2Cloud(ref.ori_rgb,
                      ref.ori_depth);


    setupSource(tmp_s);
    setupTarget(tmp_t);

    run();

    result_key = toEigen(toVector(getTransResult()));

    //cout<<"iaicp core time "<<pcl::getTime()-time_core_start;

    dvo::core::Matrix6d information;
    double loglikelihood;

    double time_llh_start = pcl::getTime();
    llhAndInfomatrix(result_key, loglikelihood, information);

    //loglikelihood = -1;
    //information = dvo::core::Matrix6d::Identity()*0.00001;

    result.Transformation = result_key.cast<double>();

    result.Information = information;
    result.LogLikelihood = loglikelihood;

    //cout<<" llh time "<<pcl::getTime()-time_llh_start;

    if(saveImage_)
    {
        //mat_source_ = cloudToImage(tmp_s);
        mat_source_ = cloudToImage(Mat2Cloud(cur.ori_rgb, cur.ori_depth));
        mat_source_.convertTo(mat_source_, CV_8UC3, 1);

        mat_target_ = cloudToImage(tmp_t);
        //mat_target_ = ref.ori_rgb;
        mat_target_.convertTo(mat_target_, CV_8UC3, 1);

        mat_trans_ = cloudToImage(tmp_s, result_key);
        mat_trans_.convertTo(mat_trans_, CV_8UC3, 1);

        cv::absdiff(mat_target_ , mat_trans_, mat_residual_);
    }
    //cout<<", iaicp total time "<<pcl::getTime()-time_start<<endl;
}

void Iaicp::setSaveImage(bool saveImage)
{
    saveImage_ = saveImage;
}

void Iaicp::getMat(Mat &m_source, Mat &m_target, Mat &m_trans, Mat &m_residual)
{
    m_source = mat_source_;
    m_target = mat_target_;
    m_trans = mat_trans_;
    m_residual = mat_residual_;
}
