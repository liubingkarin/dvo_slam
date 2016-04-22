/**
 *  This file is part of dvo.
 *
 *  Copyright 2013 Christian Kerl <christian.kerl@in.tum.de> (Technical University of Munich)
 *  For more information see <http://vision.in.tum.de/data/software/dvo>.
 *
 *  dvo is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  dvo is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with dvo.  If not, see <http://www.gnu.org/licenses/>.
 */

#include <dvo_slam/tracking_result_evaluation.h>

namespace dvo_slam
{

void TrackingResultEvaluation::add(const dvo::DenseTracker::Result& r)
{
    average_ += value(r);
    n_ += 1.0;
}

double TrackingResultEvaluation::ratioWithFirst(const dvo::DenseTracker::Result& r) const
{
    return value(r) / first_;
}

double TrackingResultEvaluation::ratioWithAverage(const dvo::DenseTracker::Result& r) const
{
    return value(r) / average_* n_;
}

TrackingResultEvaluation::TrackingResultEvaluation(double first)
{
    first_ = first;
    average_ = first_;
    n_ = 1.0;
}

double EntropyRatioTrackingResultEvaluation::value(const dvo::DenseTracker::Result& r) const
{
    return std::log(r.Information.determinant());
}

double LogLikelihoodTrackingResultEvaluation::value(const dvo::DenseTracker::Result& r) const
{
//    std::ofstream myfile;
//    myfile.open ("/home/liubing/Documents/loglikelihood.txt",std::ios::out| std::ios::app);
//    if(myfile.is_open()){
//        //std::cout<<"printing loglikelihood"<<r.LogLikelihood<<std::endl;
//        myfile<< r.LogLikelihood<<"\n";
//    }
//    else{
//        std::cout<<"file is not opened"<<std::endl;
//    }
//    myfile.close();
    return -r.LogLikelihood;
}

double NormalizedLogLikelihoodTrackingResultEvaluation::value(const dvo::DenseTracker::Result& r) const
{
    return -r.LogLikelihood / r.Statistics.Levels.back().Iterations.back().ValidConstraints;
}

} /* namespace dvo_slam */


