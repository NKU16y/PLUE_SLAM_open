#include <opencv2/core.hpp>
#include <KeyFrame.h>
#include <Frame.h>
#include <MapPoint.h>
#include <Map.h>
#include <Eigen/Core>
#include <MapLine.h>
#include <Tracking.h>
namespace PLUE_SLAM
{
class UncertaintyInInitialization
{
    // uncertainties estimation in initialization
    void StereoInitializationPointUncertainty(ORB_SLAM2::Tracking &mTracking, cv::Mat& variance, ORB_SLAM2::KeyFrame* pKFini);
    void StereoInitializationLineUncertainty(ORB_SLAM2::Tracking &mTracking, cv::Mat& variance, ORB_SLAM2::KeyFrame* pKFini);
    void StereoInitializationPointLineWithUncertainty(ORB_SLAM2::Tracking &mTracking);
};
}