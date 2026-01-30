#ifndef MAPLINE_H
#define MAPLINE_H


#include"KeyFrame.h"
#include"Frame.h"
#include"Map.h"

#include<opencv2/core/core.hpp>
#include<mutex>
#include<map>
#include<Eigen/Dense>
#include <Converter.h>
typedef Eigen::Matrix<double,6,1> Vector6d;
typedef Eigen::Matrix<double,4,1> Vector4d;

namespace ORB_SLAM2
{

class KeyFrame;
class Map;
class Frame;
    
class MapLine
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    MapLine(const Vector6d &Plk, const Eigen::Vector3d &PosStart, const Eigen::Vector3d &PosEnd,KeyFrame *pRefKF, Map *pMap);
    MapLine(const Vector6d &Plk, const Eigen::Vector3d& PosStart, const Eigen::Vector3d& PosEnd, Map* pMap, Frame* pFrame, const int &idxF);


    void SetWorldEndPoints(const Eigen::Vector3d &PosStart, const Eigen::Vector3d &PosEnd);
    void SetWorldPlk(const Vector6d &Plk);
    void SetWorldOR(const Vector4d &OR);
    void SetcovlinePlk(const cv::Mat &covPlk);    
    void SetcovlineOR(const cv::Mat &covOR);

    void GetWorldEndPoints(Eigen::Vector3d &PosStart, Eigen::Vector3d &PosEnd);
    Vector6d GetWorldPlk();
    Vector4d GetWorldOR();
    cv::Mat GetcovlinePlk();
    cv::Mat GetcovlineOR();

    Eigen::Vector3d GetNormal();
    KeyFrame* GetReferenceKeyFrame();

    float GetLength();

    std::map<KeyFrame*, size_t> GetObservations();
    int Observations();

    void AddObservation(KeyFrame *pKF, size_t idx);
    void EraseObservation(KeyFrame *pKF);

    int GetIndexInKeyFrame(KeyFrame* pKF);
    bool IsInKeyFrame(KeyFrame* pKF);

    void SetBadFlag();
    bool isBad();

    void Replace(MapLine*& pML);   
    MapLine* GetReplaced();

    void IncreaseVisible(int n=1);
    void IncreaseFound(int n=1);
    float GetFoundRatio();
    inline int GetFound(){
        return mnFound;
    }

    void ComputeDistinctiveDescriptors();

    cv::Mat GetDescriptor();

    void UpdateNormalAndDepth();
    void UdateLength(); 

    float GetMinDistanceInvariance();
    float GetMaxDistanceInvariance();

    static long unsigned int GetCurrentMaxId();   

public:
    long unsigned int mnId;
    static long unsigned int nNextId;
    const long int mnFirstKFid; //创建该MapLine的关键帧ID
    const long int mnFirstFrame;    //创建该MapLine的帧ID，每一个关键帧都有一个帧ID
    int nObs;

    float mTrackProjStartX = -1;
    float mTrackProjStartY = -1;
    float mTrackStartDepth = -1;

    float mTrackProjEndX = -1;
    float mTrackProjEndY = -1;
    float mTrackEndDepth = -1;

    // TrackLocalMap - UpdateLocalLines中防止将MapLines重复添加至mvpLocalMapLines的标记
    long unsigned int mnTrackReferenceForFrame;
    long unsigned int mnLastFrameSeen;

    bool mbTrackInView = false, mbTrackInViewR = false;


    // Variables used by local mapping
    long unsigned int mnBALocalForKF;
    long unsigned int mnFuseCandidateForKF;

    // Variables used by loop closing
    long unsigned int mnLoopLineForKF;
    long unsigned int mnCorrectedByKF;
    long unsigned int mnCorrectedReference;

    Eigen::Vector3d mPosStartGBA;
    Eigen::Vector3d mPosEndGBA;
    long unsigned int mnBAGlobalForKF;

    static std::mutex mGlobalMutex;
    
    bool iftriangulation=false;
    bool ifFused;

protected:

    // Position in absolute coordinates
    Eigen::Vector3d mWorldPosStart; //  3x1 mat
    Eigen::Vector3d mWorldPosEnd;   //  3x1 mat
    Vector6d mPosPlk;
    Vector4d mPosOR;
    cv::Mat mCovlinePlk;
    cv::Mat mCovlineOR;
    float mfLength; // [m]

    // KeyFrames observing the line and associated index in keyframe
    std::map<KeyFrame*, size_t> mObservations;   //观测到该MapLine的KF和该MapLine在KF中的索引


    // Mean viewing direction
    Eigen::Vector3d mNormalVector;

    // Best descriptor to fast matching
    cv::Mat mDescriptor;

    KeyFrame* mpRefKF;  //参考关键帧

    //Tracking counters
    int mnVisible;
    int mnFound;

    // Bad flag , we don't currently erase MapPoint from memory
    bool mbBad;
    MapLine* mpReplaced;

    // Scale invariance distances
    float mfMinDistance;
    float mfMaxDistance;

    Map* mpMap;

    std::mutex mMutexPos;
    std::mutex mMutexEndPoints;
    std::mutex mMutexPosPlk;
    std::mutex mMutexPosOR;
    std::mutex mMutexcovPlk;
    std::mutex mMutexcovOR;
    std::mutex mMutexFeatures;
};


}

#endif // MAP_LINE_H