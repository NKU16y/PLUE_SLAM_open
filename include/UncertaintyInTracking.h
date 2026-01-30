#include <opencv2/core.hpp>
#include <KeyFrame.h>
#include <Tracking.h>
#include "MapLine.h"
namespace PLUE_SLAM
{
class UncertaintyInTracking
{
    // uncertainties estimation in Tracking
    void ComputePointMatrixH4(cv::Mat &J_pose, cv::Mat &H4, ORB_SLAM2::Tracking &mTracking);
    void ComputeLineMatrixH7(cv::Mat &J_pose, cv::Mat &H4, ORB_SLAM2::Tracking &mTracking);
    void ComputeUncertaintyFromPoints(cv::Mat &J_pose,cv::Mat &H4,cv::Mat &H7,cv::Mat &pose_uncertainty_point, ORB_SLAM2::Tracking &mTracking);
    void ComputeUncertaintyFromLines(cv::Mat &Jl, cv::Mat &H4, cv::Mat &H7, cv::Mat &pose_uncertainty_line,ORB_SLAM2::Tracking &mTracking);

    // uncertainties estimation in feature creation in Tracking
    void CreateNewMapPoints(ORB_SLAM2::KeyFrame* pKF,cv::Mat &variance, ORB_SLAM2::Tracking &mTracking);
    void CreateNewMapLines(ORB_SLAM2::KeyFrame* pKF,cv::Mat &variance, ORB_SLAM2::Tracking &mTracking);
};
}