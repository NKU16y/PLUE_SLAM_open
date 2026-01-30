#include "MapLine.h"
#include "LineMatcher.h" // line matcher  for example: LBD

#include <Eigen/Core>
#include <Eigen/Dense>

#include<mutex>

typedef Eigen::Matrix<double,6,1> Vector6d;

namespace ORB_SLAM2
{

long unsigned int MapLine::nNextId=0;
mutex MapLine::mGlobalMutex;

MapLine::MapLine(const Vector6d &Plk, const Eigen::Vector3d &PosStart, const Eigen::Vector3d &PosEnd,KeyFrame *pRefKF, Map *pMap):
    mnFirstKFid(pRefKF->mnId), mnFirstFrame(pRefKF->mnFrameId), nObs(0), mnTrackReferenceForFrame(0),
    mnLastFrameSeen(0), mnBALocalForKF(0), mnFuseCandidateForKF(0), mnLoopLineForKF(0), mnCorrectedByKF(0),
    mnCorrectedReference(0), mnBAGlobalForKF(0), mpRefKF(pRefKF), mnVisible(1), mnFound(1), mbBad(false),
    mpReplaced(static_cast<MapLine*>(NULL)), mfMinDistance(0), mfMaxDistance(0),mpMap(pMap)
{
    SetWorldEndPoints(PosStart, PosEnd);

    SetWorldPlk(Plk);

    UdateLength();

    mNormalVector.setZero();

    mbTrackInViewR = false;
    mbTrackInView = false;

    unique_lock<mutex> lock(mpMap->mMutexLineCreation); // added mutexLineCreation
    mnId=nNextId++;
}

MapLine::MapLine(const Vector6d &Plk, const Eigen::Vector3d& PosStart, const Eigen::Vector3d& PosEnd, Map* pMap, Frame* pFrame, const int &idxF):
    mnFirstKFid(-1), mnFirstFrame(pFrame->mnId), nObs(0), mnTrackReferenceForFrame(0), mnLastFrameSeen(0),
    mnBALocalForKF(0), mnFuseCandidateForKF(0),mnCorrectedByKF(0),
    mnCorrectedReference(0), mnBAGlobalForKF(0), mpRefKF(static_cast<KeyFrame*>(NULL)), mnVisible(1),
    mnFound(1), mbBad(false), mpReplaced(NULL), mpMap(pMap)
{
    SetWorldEndPoints(PosStart, PosEnd); 
    
    SetWorldPlk(Plk); 

    UdateLength();

    Eigen::Vector3d p3DMiddle = 0.5*(mWorldPosStart+mWorldPosEnd);

    Eigen::Vector3d Ow;

    Ow = Converter::toVector3d(pFrame->GetCameraCenter());

    mNormalVector = p3DMiddle - Ow;

    mNormalVector.normalize();

    float distStart = (PosStart - Ow).norm();
    float distEnd   = (PosEnd - Ow).norm();

    mfMaxDistance = std::max(distStart,distEnd);
    mfMinDistance = std::min(distStart,distEnd);

    pFrame->mLineDescriptors.row(idxF).copyTo(mDescriptor); // added line descriptors

    unique_lock<mutex> lock(mpMap->mMutexLineCreation); // added line creation mutex
    mnId=nNextId++;
}

void MapLine::SetWorldEndPoints(const Eigen::Vector3d &PosStart, const Eigen::Vector3d &PosEnd)
{
    unique_lock<mutex> lock2(mGlobalMutex);
    unique_lock<mutex> lock(mMutexEndPoints);
    
    mWorldPosStart = PosStart;  
    mWorldPosEnd = PosEnd;    
}

void MapLine::SetWorldPlk(const Vector6d &Plk)
{
    unique_lock<mutex> lock2(mGlobalMutex);
    unique_lock<mutex> lock(mMutexPosPlk);
    mPosPlk = Plk;
}

void MapLine::SetWorldOR(const Vector4d &OR)
{
    unique_lock<mutex> lock2(mGlobalMutex);
    unique_lock<mutex> lock(mMutexPosOR);
    mPosOR =  OR;
}

void MapLine::SetcovlinePlk(const cv::Mat &covPlk)
{
    unique_lock<mutex> lock2(mGlobalMutex);
    unique_lock<mutex> lock(mMutexcovPlk);
    covPlk.copyTo(mCovlinePlk);
}

void MapLine::SetcovlineOR(const cv::Mat &covOR)
{
    unique_lock<mutex> lock2(mGlobalMutex);
    unique_lock<mutex> lock(mMutexcovOR);
    covOR.copyTo(mCovlineOR);
}
void MapLine::GetWorldEndPoints(Eigen::Vector3d &PosStart, Eigen::Vector3d &PosEnd)
{
    unique_lock<mutex> lock(mMutexEndPoints);
    PosStart = mWorldPosStart;    
    PosEnd = mWorldPosEnd;        
}

Vector6d MapLine::GetWorldPlk()
{
    unique_lock<mutex> lock(mMutexPosPlk);
    return mPosPlk;
}

Vector4d MapLine::GetWorldOR()
{
    unique_lock<mutex> lock(mMutexPosOR);
    return mPosOR;
}

cv::Mat MapLine::GetcovlinePlk()
{
    unique_lock<mutex> lock(mMutexcovPlk);
    return mCovlinePlk.clone();
}

cv::Mat MapLine::GetcovlineOR()
{
    unique_lock<mutex> lock(mMutexcovOR);
    return mCovlineOR.clone();
}

Eigen::Vector3d MapLine::GetNormal()
{
    unique_lock<mutex> lock(mMutexPos);
    return mNormalVector;
}

float MapLine::GetLength()
{
    unique_lock<mutex> lock(mMutexPos);
    return mfLength;    
}

KeyFrame* MapLine::GetReferenceKeyFrame()
{
    unique_lock<mutex> lock(mMutexFeatures);
    return mpRefKF;
}

void MapLine::AddObservation(KeyFrame *pKF, size_t idx)
{
    unique_lock<mutex> lock(mMutexFeatures);
    if(mObservations.count(pKF))
        return;
    mObservations[pKF]=idx;

    nObs+=2;     

}

void MapLine::EraseObservation(KeyFrame *pKF)
{
    bool bBad = false;
    {
        unique_lock<mutex> lock(mMutexFeatures);
        if(mObservations.count(pKF))
        {
            int idx = mObservations[pKF];
            if((pKF->mvuRightLineStart[idx]>=0)&&(pKF->mvuRightLineEnd[idx]>=0)) // added line endPoints
                nObs-=2;
            else
                nObs--;

            mObservations.erase(pKF);

            if(mpRefKF==pKF)
                mpRefKF=mObservations.begin()->first;

            if(nObs<=2)
                bBad=true;
        }
    }

    if(bBad)
        SetBadFlag();
}

std::map<KeyFrame*, size_t>  MapLine::GetObservations()
{
    unique_lock<mutex> lock(mMutexFeatures);
    return mObservations;
}

int MapLine::Observations()
{
    unique_lock<mutex> lock(mMutexFeatures);
    return nObs;
}

void MapLine::ComputeDistinctiveDescriptors()
{
    // Retrieve all observed descriptors
    vector<cv::Mat> vDescriptors;

    map<KeyFrame*,size_t> observations;

    {
        unique_lock<mutex> lock1(mMutexFeatures);
        if(mbBad)
            return;
        observations=mObservations;
    }

    if(observations.empty())
        return;

    vDescriptors.reserve(observations.size());

    for(map<KeyFrame*,size_t>::iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
    {
        KeyFrame* pKF = mit->first;

        if(!pKF->isBad())
            vDescriptors.push_back(pKF->mLineDescriptors.row(mit->second)); // added line descriptors
    }

    if(vDescriptors.empty())
            return;
    
    const size_t N = vDescriptors.size();

    float Distances[N][N];
    for(size_t i=0;i<N;i++)
    {
        Distances[i][i]=0;
        for(size_t j=i+1;j<N;j++)
        {
            // int distij = ORBmatcher::DescriptorDistance(vDescriptors[i],vDescriptors[j]);
            //const int distij = cv::norm(vDescriptors[i],vDescriptors[j],cv::NORM_HAMMING);
            int distij = LineMatcher::DescriptorDistance(vDescriptors[i],vDescriptors[j]);
            Distances[i][j]=distij;
            Distances[j][i]=distij;
        }
    }

    // Take the descriptor with least median distance to the rest
    int BestMedian = INT_MAX;
    int BestIdx = 0;
    for(size_t i=0;i<N;i++)
    {
        vector<int> vDists(Distances[i],Distances[i]+N);
        sort(vDists.begin(),vDists.end());
        int median = vDists[0.5*(N-1)];

        if(median<BestMedian)
        {
            BestMedian = median;
            BestIdx = i;
        }
    }

    {
        unique_lock<mutex> lock(mMutexFeatures);
        mDescriptor = vDescriptors[BestIdx].clone();
    }

}

int MapLine::GetIndexInKeyFrame(KeyFrame *pKF)
{
    unique_lock<mutex> lock(mMutexFeatures);
    if(mObservations.count(pKF))
        return mObservations[pKF];
    else
        return -1;
}

cv::Mat MapLine::GetDescriptor()
{
    unique_lock<mutex> lock(mMutexFeatures);
    return mDescriptor.clone();
}

bool MapLine::IsInKeyFrame(KeyFrame* pKF)
{
    unique_lock<mutex> lock(mMutexFeatures);
    return (mObservations.count(pKF));
}

void MapLine::UpdateNormalAndDepth()
{
    std::map<KeyFrame*,size_t> observations;
    KeyFrame* pRefKF;
    Eigen::Vector3d p3DStart;
    Eigen::Vector3d p3DEnd;

    {
        unique_lock<mutex> lock1(mMutexFeatures);
        unique_lock<mutex> lock2(mMutexPos);
        if(mbBad)
            return;
        observations = mObservations;
        pRefKF = mpRefKF;
        p3DStart = mWorldPosStart;
        p3DEnd   = mWorldPosEnd;
    }

    if(observations.empty())
        return;
    
    UdateLength();


    Eigen::Vector3d Plk_v = p3DEnd - p3DStart;

    Eigen::Vector3d Plk_n = Plk_v.cross(p3DStart);
    mPosPlk.segment<3>(0) = Plk_n;
    mPosPlk.segment<3>(3) = Plk_v;

    const Eigen::Vector3d p3DMiddle = 0.5*(p3DStart + p3DEnd);
    Eigen::Vector3d normal = Eigen::Vector3d::Zero();
    int n=0;
    for(map<KeyFrame*,size_t>::iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
    {
        KeyFrame* pKF = mit->first;

        Eigen::Vector3d Owi = Converter::toVector3d(pKF->GetCameraCenter());
        Eigen::Vector3d normali = p3DMiddle - Owi;
        normal = normal + normali/normali.norm();
        n++;

    }

    const Eigen::Vector3d Ow = Converter::toVector3d(pRefKF->GetCameraCenter());
    float distStart = (p3DStart - Ow).norm();
    float distEnd   = (p3DEnd - Ow).norm();
    //float distMiddle = cv::norm(p3DMiddle - Ow);

    {
        unique_lock<mutex> lock3(mMutexPos);
	    mfMaxDistance = std::max(distStart,distEnd);
        mfMinDistance = std::min(distStart,distEnd);

        mNormalVector = normal/n;
        mNormalVector.normalize();
    }
}

void MapLine::SetBadFlag()
{
    map<KeyFrame*, size_t> obs;
    {
        unique_lock<mutex> lock1(mMutexFeatures);
        unique_lock<mutex> lock2(mMutexPos);
        unique_lock<mutex> lock3(mMutexEndPoints);
        unique_lock<mutex> lock4(mMutexPosPlk);
        unique_lock<mutex> lock5(mMutexPosOR);
        unique_lock<mutex> lock6(mMutexcovPlk);
        unique_lock<mutex> lock7(mMutexcovOR);

        mbBad=true;
        obs = mObservations;    
        mObservations.clear();  
    }

    for(map<KeyFrame*, size_t>::iterator mit=obs.begin(), mend=obs.end(); mit!=mend; mit++)
    {
        KeyFrame* pKF = mit->first;
        pKF->EraseMapLineMatch(mit->second); 
    }

    mpMap->EraseMapLine(this);  
}

bool MapLine::isBad()
{
    unique_lock<mutex> lock(mMutexFeatures);
    unique_lock<mutex> lock2(mMutexPos);
    unique_lock<mutex> lock3(mMutexEndPoints);
    unique_lock<mutex> lock4(mMutexPosPlk);
    unique_lock<mutex> lock5(mMutexPosOR);
    unique_lock<mutex> lock6(mMutexcovPlk);
    unique_lock<mutex> lock7(mMutexcovOR);

        return mbBad;
}

void MapLine::IncreaseVisible(int n)
{
    unique_lock<mutex> lock(mMutexFeatures);
    mnVisible+=n;
}

void MapLine::IncreaseFound(int n)
{
    unique_lock<mutex> lock(mMutexFeatures);
    mnFound+=n;
}

float MapLine::GetFoundRatio()
{
    unique_lock<mutex> lock(mMutexFeatures);
    return static_cast<float>(mnFound)/mnVisible;
}

MapLine* MapLine::GetReplaced()
{
    unique_lock<mutex> lock1(mMutexFeatures);
    unique_lock<mutex> lock2(mMutexPos);
    unique_lock<mutex> lock3(mMutexEndPoints);
    unique_lock<mutex> lock4(mMutexPosPlk);
    unique_lock<mutex> lock5(mMutexPosOR);
    unique_lock<mutex> lock6(mMutexcovPlk);
    unique_lock<mutex> lock7(mMutexcovOR);
    return mpReplaced;
}

void MapLine::Replace(MapLine*& pML)
{
    if(pML->mnId==this->mnId)
        return;

    int nvisible, nfound;
    map<KeyFrame*,size_t> obs;
    {
        unique_lock<mutex> lock1(mMutexFeatures);
        unique_lock<mutex> lock2(mMutexPos);
        unique_lock<mutex> lock3(mMutexEndPoints);
        unique_lock<mutex> lock4(mMutexPosPlk);
        unique_lock<mutex> lock5(mMutexPosOR);
        unique_lock<mutex> lock6(mMutexcovPlk);
        unique_lock<mutex> lock7(mMutexcovOR);
        obs=mObservations;
        mObservations.clear();
        mbBad=true;
        nvisible = mnVisible;
        nfound = mnFound;        
        mpReplaced = pML;
    }

    for(map<KeyFrame*,size_t>::iterator mit=obs.begin(), mend=obs.end(); mit!=mend; mit++)
    {
        // Replace measurement in keyframe
        KeyFrame* pKF = mit->first;

        if(!pML->IsInKeyFrame(pKF))
        {
            pKF->ReplaceMapLineMatch(mit->second, pML); // added ReplaceMapLineMatch
            pML->AddObservation(pKF,mit->second);
        }
        else
        {
            pKF->EraseMapLineMatch(mit->second);  // added EraseMapLineMatch
        }
    }

    pML->IncreaseFound(nfound);
    pML->IncreaseVisible(nvisible);
    pML->ComputeDistinctiveDescriptors();

    mpMap->EraseMapLine(this); // added EraseMapLine
}

void MapLine::UdateLength()
{
    mfLength = (mWorldPosStart - mWorldPosEnd).norm();
}


float MapLine::GetMinDistanceInvariance()
{
    unique_lock<mutex> lock(mMutexPos);
    return 0.8f*mfMaxDistance; // 0.8 ~= 1/1.2
    //return 0.7f*mfMinDistance; // 0.7 ~= 1/1.4
}

float MapLine::GetMaxDistanceInvariance()
{
    unique_lock<mutex> lock(mMutexPos);
    return 1.2f*mfMaxDistance;
    //return 1.4f*mfMaxDistance;
}

long unsigned int MapLine::GetCurrentMaxId()
{
    //unique_lock<mutex> lock(mpMap->mMutexPointCreation);
    return nNextId;    
}

}