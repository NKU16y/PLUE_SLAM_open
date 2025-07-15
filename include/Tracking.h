
namespace ORB_SLAM2
{   
class Tracking
{  
    // uncertainties estimation in initialization
    void StereoInitializationPointUncertainty(KeyFrame* pKFini, cv::Mat& variance);
    void StereoInitializationLineUncertainty(KeyFrame* pKFini, cv::Mat& variance);
    void StereoInitializationPointLineWithUncertainty();

    // uncertainties estimation in Tracking
    void ComputePointMatrixH4(cv::Mat &J_pose, cv::Mat &H4);
    void ComputeLineMatrixH7(cv::Mat &J_pose, cv::Mat &H4);
    void ComputeUncertaintyFromPoints(cv::Mat &J_pose,cv::Mat &H4,cv::Mat &H7,cv::Mat &pose_uncertainty_point);
    void ComputeUncertaintyFromLines(cv::Mat &Jl, cv::Mat &H4, cv::Mat &H7, cv::Mat &pose_uncertainty_line);
    void TrackPointLineWithUncertainty();

    // uncertainties estimation in feature creation in Tracking
    void CreateNewMapPoints(KeyFrame* pKF,cv::Mat &variance);
    void CreateNewMapLines(KeyFrame* pKF,cv::Mat &variance);
    void CreateNewKeyFrameWithUncertainty();
}
}