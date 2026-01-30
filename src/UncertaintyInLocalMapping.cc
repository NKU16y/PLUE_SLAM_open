#include <opencv2/core.hpp>
#include <KeyFrame.h>
#include <Frame.h>
#include <MapPoint.h>
#include <Map.h>
#include <ORBmatcher.h>
#include <LocalMapping.h>
// add in LocalMapping.h and LocalMapping.cc
namespace ORB_SLAM2
{

void LocalMapping::CreateNewMapPointsUncertainty(const vector<ORB_SLAM2::KeyFrame*> &vpNeighKFs, ORB_SLAM2::LocalMapping &mLM)
{
    ORB_SLAM2::ORBmatcher matcher(0.6,false);
    cv::Mat Rcw1 = mpCurrentKeyFrame->GetRotation();
    cv::Mat tcw1 = mpCurrentKeyFrame->GetTranslation();
    cv::Mat Tcw1 = cv::Mat::eye(4,4,CV_32F);
    Rcw1.copyTo(Tcw1.rowRange(0,3).colRange(0,3));
    tcw1.copyTo(Tcw1.rowRange(0,3).col(3));
    cv::Mat Ow1 = mpCurrentKeyFrame->GetCameraCenter();

    const float &fx1 = mpCurrentKeyFrame->fx;
    const float &fy1 = mpCurrentKeyFrame->fy;
    const float &cx1 = mpCurrentKeyFrame->cx;
    const float &cy1 = mpCurrentKeyFrame->cy;
    const float &invfx1 = mpCurrentKeyFrame->invfx;
    const float &invfy1 = mpCurrentKeyFrame->invfy;
    const float ratioFactor = 1.5f*mpCurrentKeyFrame->mfScaleFactor;

    int nnew=0;
    // 算位姿的J矩阵
    Eigen::Matrix3d Rcw_eigen = Converter::toMatrix3d(Rcw1);    //当前帧旋转矩阵3*3
    Eigen::Vector3d tcw_eigen(tcw1.at<float>(0,0), tcw1.at<float>(1,0),tcw1.at<float>(2,0));


    Sophus::SE3 SE3_Rt(Rcw_eigen, tcw_eigen);

    Eigen::Matrix<double, 6, 1> se3 = SE3_Rt.log(); 

    Eigen::Vector3d phi = se3.topRows(3);
    Eigen::Vector3d rho = se3.bottomRows(3);

    Eigen::Matrix<double, 6, 6> J_pose;
    J_pose.setZero();

    J_pose.topLeftCorner(3,3) = hat(phi);
    J_pose.topRightCorner(3,1) = hat(rho);
    J_pose.bottomRightCorner(3,3) = hat(phi);
    J_pose = Eigen::MatrixXd::Identity(6, 6) +  0.5  * J_pose;
    cv::Mat J_pose_f = Converter::toCvMat(J_pose);
    cv::Mat covU = 1.5 * cv::Mat::eye(3,3,CV_32F);
    cv::Mat K = mpCurrentKeyFrame->mK.clone();
    cv::Mat covXc = K.inv() * covU * K.inv().t();
    cv::Mat varianceKF1 = mpCurrentKeyFrame->GetDepthVarMat();
    cv::Mat covPose1 = mpCurrentKeyFrame->GetCovpose();

    // 得到伴随矩阵
    cv::Mat Twc1 = Tcw1.inv();
    cv::Mat AdTwc1 = cv::Mat::zeros(6,6,CV_32F);
    cv::Mat Rwc1 = cv::Mat::zeros(3,3,CV_32F);
    cv::Mat twc1 = cv::Mat::zeros(3,1,CV_32F);
    cv::Mat skewtwc1 = cv::Mat::zeros(3,3,CV_32F);
    cv::Mat skewtwcRwc1 = cv::Mat::zeros(3,3,CV_32F);
    Twc1.colRange(0,3).rowRange(0,3).copyTo(Rwc1);
    Twc1.rowRange(0,3).col(3).copyTo(twc1);
    skewtwc1 = skew(twc1);
    skewtwcRwc1 = skewtwc1 * Rwc1;
    Rwc1.copyTo(AdTwc1.rowRange(0,3).colRange(0,3));
    Rwc1.copyTo(AdTwc1.rowRange(3,6).colRange(3,6));
    skewtwcRwc1.copyTo(AdTwc1.rowRange(0,3).colRange(3,6));
    // J_poseinv
    Eigen::Matrix3d Rwc1_eigen = Converter::toMatrix3d(Rwc1);    
    Eigen::Vector3d twc1_eigen = Converter::toVector3d(twc1);  
    Sophus::SE3 SE3_Rt1_inv(Rwc1_eigen,twc1_eigen);
    Eigen::Matrix<double, 6, 1> se31 = SE3_Rt1_inv.log(); 
    Eigen::Vector3d phi1 = se31.topRows(3);
    Eigen::Vector3d rho1 = se31.bottomRows(3);
    Eigen::Matrix<double,6,6> J_poseinv1;
    J_poseinv1.setZero();

    J_poseinv1.topLeftCorner(3,3) = hat(phi1);
    J_poseinv1.topRightCorner(3,1) = hat(rho1);
    J_poseinv1.bottomRightCorner(3,3) = hat(phi1);
    J_poseinv1 = Eigen::MatrixXd::Identity(6, 6) +  0.5  * J_poseinv1;

    cv::Mat J_poseinv1_f = Converter::toCvMat(J_poseinv1);
    // Search matches with epipolar restriction and triangulate
    for(size_t i=0; i<vpNeighKFs.size(); i++)
    {
        if(i>0 && CheckNewKeyFrames())
            return;
        
        KeyFrame* pKF2 = vpNeighKFs[i];
        cv::Mat varianceKF2 = pKF2->GetDepthVarMat();
        // Check first that baseline is not too short
        cv::Mat Ow2 = pKF2->GetCameraCenter();
        cv::Mat vBaseline = Ow2-Ow1;
        const float baseline = cv::norm(vBaseline);

        if(!mbMonocular)
        {
            if(baseline<pKF2->mb)
            continue;
        }
        else
        {
            const float medianDepthKF2 = pKF2->ComputeSceneMedianDepth(2);
            const float ratioBaselineDepth = baseline/medianDepthKF2;

            if(ratioBaselineDepth<0.01)
                continue;
        }

        // Compute Fundamental Matrix
        cv::Mat F12 = ComputeF12(mpCurrentKeyFrame,pKF2);

        // Search matches that fullfil epipolar constraint
        vector<pair<size_t,size_t> > vMatchedIndices;
        matcher.SearchForTriangulation(mpCurrentKeyFrame,pKF2,F12,vMatchedIndices,false);

        cv::Mat Rcw2 = pKF2->GetRotation();
        cv::Mat tcw2 = pKF2->GetTranslation();
        cv::Mat Tcw2 = cv::Mat::eye(4,4,CV_32F);
        Rcw2.copyTo(Tcw2.rowRange(0,3).colRange(0,3));
        tcw2.copyTo(Tcw2.rowRange(0,3).col(3));

        const float &fx2 = pKF2->fx;
        const float &fy2 = pKF2->fy;
        const float &cx2 = pKF2->cx;
        const float &cy2 = pKF2->cy;
        const float &invfx2 = pKF2->invfx;
        const float &invfy2 = pKF2->invfy;
        cv::Mat covPose2 = pKF2->GetCovpose();
       
        // 得到伴随矩阵
        cv::Mat Twc2 = Tcw2.inv();
        cv::Mat AdTwc2 = cv::Mat::zeros(6,6,CV_32F);
        cv::Mat Rwc2 = cv::Mat::zeros(3,3,CV_32F);
        cv::Mat twc2 = cv::Mat::zeros(3,1,CV_32F);
        cv::Mat skewtwc2 = cv::Mat::zeros(3,3,CV_32F);
        cv::Mat skewtwcRwc2 = cv::Mat::zeros(3,3,CV_32F);
        Twc2.colRange(0,3).rowRange(0,3).copyTo(Rwc2);
        Twc2.rowRange(0,3).col(3).copyTo(twc2);
        skewtwc2 = skew(twc2);
        skewtwcRwc2 = skewtwc2 * Rwc2;
        Rwc2.copyTo(AdTwc2.rowRange(0,3).colRange(0,3));
        Rwc2.copyTo(AdTwc2.rowRange(3,6).colRange(3,6));
        skewtwcRwc2.copyTo(AdTwc2.rowRange(0,3).colRange(3,6));
        // J_poseinv
        Eigen::Matrix3d Rwc2_eigen = Converter::toMatrix3d(Rwc2);    
        Eigen::Vector3d twc2_eigen = Converter::toVector3d(twc2);  
        Sophus::SE3 SE3_Rt2_inv(Rwc2_eigen,twc2_eigen);
        Eigen::Matrix<double, 6, 1> se32 = SE3_Rt2_inv.log(); 
        Eigen::Vector3d phi2 = se32.topRows(3);
        Eigen::Vector3d rho2 = se32.bottomRows(3);
        Eigen::Matrix<double,6,6> J_poseinv2;
        J_poseinv2.setZero();

        J_poseinv2.topLeftCorner(3,3) = hat(phi2);
        J_poseinv2.topRightCorner(3,1) = hat(rho2);
        J_poseinv2.bottomRightCorner(3,3) = hat(phi2);
        J_poseinv2 = Eigen::MatrixXd::Identity(6, 6) +  0.5  * J_poseinv2;

        cv::Mat J_poseinv2_f = Converter::toCvMat(J_poseinv2);
        // Triangulate each match
        const int nmatches = vMatchedIndices.size();
        for(int ikp=0; ikp<nmatches; ikp++)
        {
            bool iftriangulation = false;
            int depthSource = 0;
            const int &idx1 = vMatchedIndices[ikp].first;
            const int &idx2 = vMatchedIndices[ikp].second;

            const cv::KeyPoint &kp1 = mpCurrentKeyFrame->mvKeysUn[idx1];
            const float kp1_ur=mpCurrentKeyFrame->mvuRight[idx1];
            bool bStereo1 = kp1_ur>=0;

            const cv::KeyPoint &kp2 = pKF2->mvKeysUn[idx2];
            const float kp2_ur = pKF2->mvuRight[idx2];
            bool bStereo2 = kp2_ur>=0;

            // Check parallax between rays
            cv::Mat xn1 = (cv::Mat_<float>(3,1) << (kp1.pt.x-cx1)*invfx1, (kp1.pt.y-cy1)*invfy1, 1.0);
            cv::Mat xn2 = (cv::Mat_<float>(3,1) << (kp2.pt.x-cx2)*invfx2, (kp2.pt.y-cy2)*invfy2, 1.0);

            cv::Mat ray1 = Rwc1*xn1;
            cv::Mat ray2 = Rwc2*xn2;
            const float cosParallaxRays = ray1.dot(ray2)/(cv::norm(ray1)*cv::norm(ray2));

            float cosParallaxStereo = cosParallaxRays+1;
            float cosParallaxStereo1 = cosParallaxStereo;
            float cosParallaxStereo2 = cosParallaxStereo;

            if(bStereo1)
                cosParallaxStereo1 = cos(2*atan2(mpCurrentKeyFrame->mb/2,mpCurrentKeyFrame->mvDepth[idx1]));
            else if(bStereo2)
                cosParallaxStereo2 = cos(2*atan2(pKF2->mb/2,pKF2->mvDepth[idx2]));
            
            cosParallaxStereo = min(cosParallaxStereo1,cosParallaxStereo2);

            cv::Mat x3D;
            if(cosParallaxRays<cosParallaxStereo && cosParallaxRays>0 && (bStereo1 || bStereo2 || cosParallaxRays<0.9998))
            {
                // Linear Triangulation Method
                cv::Mat A(4,4,CV_32F);
                A.row(0) = xn1.at<float>(0)*Tcw1.row(2)-Tcw1.row(0);
                A.row(1) = xn1.at<float>(1)*Tcw1.row(2)-Tcw1.row(1);
                A.row(2) = xn2.at<float>(0)*Tcw2.row(2)-Tcw2.row(0);
                A.row(3) = xn2.at<float>(1)*Tcw2.row(2)-Tcw2.row(1);

                cv::Mat w,u,vt;
                cv::SVD::compute(A,w,u,vt,cv::SVD::MODIFY_A| cv::SVD::FULL_UV);

                x3D = vt.row(3).t();

                if(x3D.at<float>(3)==0)
                    continue;

                // Euclidean coordinates
                x3D = x3D.rowRange(0,3)/x3D.at<float>(3);
                iftriangulation = true;
            }

            else if(bStereo1 && cosParallaxStereo1<cosParallaxStereo2)
            {
                x3D = mpCurrentKeyFrame->UnprojectStereo(idx1);
                iftriangulation = false;
                depthSource = 1;
            }

            else if(bStereo2 && cosParallaxStereo2<cosParallaxStereo1)
            {
                x3D = pKF2->UnprojectStereo(idx2);
                iftriangulation = false;
                depthSource = 2;
            }
            else
                continue;  //No stereo and very low parallax
            
            cv::Mat x3Dt = x3D.t();

            //Check triangulation in front of cameras
            float z1 = Rcw1.row(2).dot(x3Dt)+tcw1.at<float>(2);
            if(z1<=0)
                continue;
            
            float z2 = Rcw2.row(2).dot(x3Dt)+tcw2.at<float>(2);
            if(z2<=0)
                continue;

            //Check reprojection error in first keyframe
            const float &sigmaSquare1 = mpCurrentKeyFrame->mvLevelSigma2[kp1.octave];
            const float x1 = Rcw1.row(0).dot(x3Dt)+tcw1.at<float>(0);
            const float y1 = Rcw1.row(1).dot(x3Dt)+tcw1.at<float>(1);
            const float invz1 = 1.0/z1;

            if(!bStereo1)
            {
                float u1 = fx1*x1*invz1+cx1;
                float v1 = fy1*y1*invz1+cy1;
                float errX1 = u1 - kp1.pt.x;
                float errY1 = v1 - kp1.pt.y;
                if((errX1*errX1+errY1*errY1)>5.991*sigmaSquare1)
                    continue;
            }
            else
            {
                float u1 = fx1*x1*invz1+cx1;
                float u1_r = u1 - mpCurrentKeyFrame->mbf*invz1;
                float v1 = fy1*y1*invz1+cy1;
                float errX1 = u1 - kp1.pt.x;
                float errY1 = v1 - kp1.pt.y;
                float errX1_r = u1_r - kp1_ur;
                if((errX1*errX1+errY1*errY1+errX1_r*errX1_r)>7.8*sigmaSquare1)
                    continue;
            }

            //Check reprojection error in second keyframe
            const float sigmaSquare2 = pKF2->mvLevelSigma2[kp2.octave];
            const float x2 = Rcw2.row(0).dot(x3Dt)+tcw2.at<float>(0);
            const float y2 = Rcw2.row(1).dot(x3Dt)+tcw2.at<float>(1);
            const float invz2 = 1.0/z2;
            if(!bStereo2)
            {
                float u2 = fx2*x2*invz2+cx2;
                float v2 = fy2*y2*invz2+cy2;
                float errX2 = u2 - kp2.pt.x;
                float errY2 = v2 - kp2.pt.y;
                if((errX2*errX2+errY2*errY2)>5.991*sigmaSquare2)
                    continue;
            }
            else
            {
                float u2 = fx2*x2*invz2+cx2;
                float u2_r = u2 - mpCurrentKeyFrame->mbf*invz2;
                float v2 = fy2*y2*invz2+cy2;
                float errX2 = u2 - kp2.pt.x;
                float errY2 = v2 - kp2.pt.y;
                float errX2_r = u2_r - kp2_ur;
                if((errX2*errX2+errY2*errY2+errX2_r*errX2_r)>7.8*sigmaSquare2)
                    continue;
            }

            //Check scale consistency
            cv::Mat normal1 = x3D-Ow1;
            float dist1 = cv::norm(normal1);

            cv::Mat normal2 = x3D-Ow2;
            float dist2 = cv::norm(normal2);

            if(dist1==0 || dist2==0)
                continue;

            const float ratioDist = dist2/dist1;
            const float ratioOctave = mpCurrentKeyFrame->mvScaleFactors[kp1.octave]/pKF2->mvScaleFactors[kp2.octave];

            if(ratioDist*ratioFactor<ratioOctave || ratioDist>ratioOctave*ratioFactor)
                continue;
            
            // Triangulation is succesfull
            MapPoint* pMP = new MapPoint(x3D,mpCurrentKeyFrame,mpMap);
            cv::Mat covXw = cv::Mat::zeros(3,3,CV_32F);
            cv::Mat pXc_U = cv::Mat::zeros(3,3,CV_32F);
            cv::Mat pXc_z = cv::Mat::zeros(3,1,CV_32F);
            const float &u1 = kp1.pt.x;
            const float &v1 = kp1.pt.y;
            const float &u2 = kp2.pt.x;
            const float &v2 = kp2.pt.y;
            // uncertainty
            if(iftriangulation)
            {
                // cov_Xw = H1.inverse() * H2.transpose() cov(Xc) H2 * H1.inverse().transpose()
                // H1i
                cv::Mat H1i = cv::Mat::zeros(3,3,CV_32F);
                cv::Mat R = Rcw1.clone();
                cv::Mat t = tcw1.clone();
                cv::Mat Ui = cv::Mat::zeros(3,1,CV_32F);
                Ui.at<float>(0,0) = u1;
                Ui.at<float>(1,0) = v1;
                Ui.at<float>(2,0) = 1.0;
                cv::Mat Xc = cv::Mat::zeros(3,1,CV_32F);
                Xc = K.inv() * Ui;
                cv::Mat skewXc = cv::Mat::zeros(3,3,CV_32F);
                skewXc = skew(Xc);
                H1i = -2*R.t()*skewXc*skewXc*R;
                // H2i
                cv::Mat H2i = cv::Mat::zeros(3,3,CV_32F);
                cv::Mat Bi = cv::Mat::zeros(3,1,CV_32F);
                cv::Mat temp = cv::Mat::zeros(3,3,CV_32F);
                Bi = R * x3D + t;
                temp = Xc.dot(Bi)*cv::Mat::eye(3,3,CV_32F) + Xc*Bi.t() - 2 * Bi*Xc.t();
                H2i = -2*(R.t() * temp).t();
                // H3i
                cv::Mat H3i = cv::Mat::zeros(6,3,CV_32F);
                cv::Mat H3it = cv::Mat::zeros(3,6,CV_32F);
                cv::Mat fuskewBiI = cv::Mat::zeros(3,6,CV_32F);
                cv::Mat fuskewBi = -skew(Bi);
                fuskewBi.rowRange(0, 3).colRange(0, 3).copyTo(fuskewBiI.rowRange(0, 3).colRange(0, 3));
                cv::Mat I33 = cv::Mat::eye(3,3,CV_32F);
                I33.rowRange(0, 3).colRange(0, 3).copyTo(fuskewBiI.rowRange(0, 3).colRange(3, 6));
                
                H3it = R.t() * skewXc * skewXc * fuskewBiI * J_pose_f;

                H3i = -2 * H3it.t();

                cv::Mat H1iinv = H1i.inv();
                covXw = cv::Mat::zeros(3,3,CV_32F);
                covXw = H1iinv * H2i.t() * covXc * H2i * H1iinv.t() + H1iinv * H3i.t() * covPose1 * H3i * H1iinv.t();
                pMP->iftriangulation = true;
                pMP->ifFused = false;
            }

            else //深度测量
            {
                pMP->iftriangulation = false;
                pMP->ifFused = true;
                if(depthSource == 1)
                {
                    int u = cvRound(u1);
                    int v = cvRound(v1);
                    if(u>=0 && u<varianceKF1.cols && v>=0 && v<varianceKF1.rows) {
                        float varz = varianceKF1.at<float>(v,u);
                        cv::Mat x3Dc = Rcw1 * x3D + tcw1;

                        float zc = x3Dc.at<float>(2,0);

                        covXc = cv::Mat::zeros(3,3,CV_32F);
                        pXc_U = cv::Mat::zeros(3,3,CV_32F);
                        pXc_U.at<float>(0,0) = zc*invfx1;
                        pXc_U.at<float>(1,1) = zc*invfy1;
                        
                        pXc_z = cv::Mat::zeros(3,1,CV_32F);
                        pXc_z.at<float>(0,0) = (u1-cx1) * invfx1;
                        pXc_z.at<float>(1,0) = (v1-cy1) * invfy1;
                        pXc_z.at<float>(2,0) = 1.0;

                        covXc = pXc_U*covU*pXc_U.t() + pXc_z * varz * pXc_z.t();
                        cv::Mat covPoseInv = cv::Mat::zeros(6,6,CV_32F);
                        covPoseInv = AdTwc1 * covPose1 * AdTwc1.t();
                        cv::Mat pXw_poseinv = cv::Mat::zeros(3,6,CV_32F);
                        const float &x = x3D.at<float>(0,0);
                        const float &y = x3D.at<float>(1,0);
                        const float &z = x3D.at<float>(2,0);
                        cv::Mat pXw_poseinv_initial = (cv::Mat_<float>(3,6) <<
                        0,  z, -y, 1, 0, 0,
                        -z,  0,  x, 0, 1, 0,
                        y, -x,  0, 0, 0, 1
                        );
                        pXw_poseinv = pXw_poseinv_initial * J_poseinv1_f;

                        covXw = Rwc1*covXc*Rwc1.t() + pXw_poseinv*covPoseInv*pXw_poseinv.t();  
                    }
                }
                else if(depthSource == 2)
                {
                    int u = cvRound(u2);
                    int v = cvRound(v2);
                    if(u>=0 && u<varianceKF2.cols && v>=0 && v<varianceKF2.rows) {
                        float varz = varianceKF2.at<float>(v,u);
                        cv::Mat x3Dc = Rcw2 * x3D + tcw2;

                        float zc = x3Dc.at<float>(2,0);

                        covXc = cv::Mat::zeros(3,3,CV_32F);
                        pXc_U = cv::Mat::zeros(3,3,CV_32F);
                        pXc_U.at<float>(0,0) = zc*invfx2;
                        pXc_U.at<float>(1,1) = zc*invfy2;

                        pXc_z = cv::Mat::zeros(3,1,CV_32F);
                        pXc_z.at<float>(0,0) = (u2-cx2) * invfx2;
                        pXc_z.at<float>(1,0) = (v2-cy2) * invfy2;
                        pXc_z.at<float>(2,0) = 1.0;

                        covXc = pXc_U*covU*pXc_U.t() + pXc_z * varz * pXc_z.t();
                        cv::Mat covPoseInv = cv::Mat::zeros(6,6,CV_32F);
                        covPoseInv = AdTwc2 * covPose2 * AdTwc2.t();
                        cv::Mat pXw_poseinv = cv::Mat::zeros(3,6,CV_32F);
                        const float &x = x3D.at<float>(0,0);
                        const float &y = x3D.at<float>(1,0);
                        const float &z = x3D.at<float>(2,0);
                        cv::Mat pXw_poseinv_initial = (cv::Mat_<float>(3,6) <<
                        0,  z, -y, 1, 0, 0,
                        -z,  0,  x, 0, 1, 0,
                        y, -x,  0, 0, 0, 1
                        );
                        pXw_poseinv = pXw_poseinv_initial * J_poseinv2_f;

                        covXw = Rwc2*covXc*Rwc2.t() + pXw_poseinv*covPoseInv*pXw_poseinv.t();
                    }
                }
            }
            pMP->SetCovtri3(covXw);

            pMP->AddObservation(mpCurrentKeyFrame,idx1);  
            pMP->AddObservation(pKF2,idx2);

            mpCurrentKeyFrame->AddMapPoint(pMP,idx1);
            pKF2->AddMapPoint(pMP,idx2);

            pMP->ComputeDistinctiveDescriptors();
            pMP->UpdateNormalAndDepth();

            mpMap->AddMapPoint(pMP);
            mlpRecentAddedMapPoints.push_back(pMP);

            nnew++;
        }
    }
}

void computeH21(KeyFrame*& pKF1, KeyFrame*& pKF2, cv::Mat& H21, cv::Mat& e2)
{
    const cv::Mat R1w = pKF1->GetRotation();
    const cv::Mat t1w = pKF1->GetTranslation();
    const cv::Mat R2w = pKF2->GetRotation();
    const cv::Mat t2w = pKF2->GetTranslation();

    const cv::Mat R21 = R2w*R1w.t();
    const cv::Mat t21 = -R21*t1w+t2w; 

    const cv::Mat &K1 = pKF1->mK;
    const cv::Mat &K2 = pKF2->mK; 

    const cv::Mat K1inv = (cv::Mat_<float>(3,3) <<  pKF1->invfx,           0,  -pKF1->cx*pKF1->invfx, 
                                                              0, pKF1->invfy,  -pKF1->cy*pKF1->invfy, 
                                                              0,           0,                     1.);                                                
    
    const cv::Mat K2inv = (cv::Mat_<float>(3,3) <<  pKF2->invfx,           0,  -pKF2->cx*pKF2->invfx, 
                                                              0, pKF2->invfy,  -pKF2->cy*pKF2->invfy, 
                                                              0,           0,                     1.);
    
    const cv::Mat R21xK1inv = R21*K1inv;
    H21 = K2*R21xK1inv;
    e2  = K2*t21;
}

void LocalMapping::CreateNewMapLinesUncertainty(const vector<KeyFrame*> &vpNeighKFs)
{
    std::unique_ptr<LineMatcher> pLineMatcher;
    pLineMatcher.reset( new LineMatcher(0.6,false) );

    cv::Mat Rcw1 = mpCurrentKeyFrame->GetRotation();
    cv::Mat tcw1 = mpCurrentKeyFrame->GetTranslation();
    cv::Mat Tcw1 = cv::Mat::eye(4,4,CV_32F);
    Rcw1.copyTo(Tcw1.rowRange(0,3).colRange(0,3));
    tcw1.copyTo(Tcw1.rowRange(0,3).col(3));
    cv::Mat Ow1 = mpCurrentKeyFrame->GetCameraCenter();

    Eigen::Matrix3d Rcw1_eg = Converter::toMatrix3d(Rcw1);
    Eigen::Matrix3d Rwc1_eg = Rcw1_eg.transpose();
    Eigen::Vector3d tcw1_eg = Converter::toVector3d(tcw1);
    Eigen::Vector3d twc1_eg = -Rwc1_eg*tcw1_eg;
    Eigen::Vector3d Ow1_eg = Converter::toVector3d(Ow1);

    const float &fx1 = mpCurrentKeyFrame->fx;
    const float &fy1 = mpCurrentKeyFrame->fy;
    const float &cx1 = mpCurrentKeyFrame->cx;
    const float &cy1 = mpCurrentKeyFrame->cy;
    const float &invfx1 = mpCurrentKeyFrame->invfx;
    const float &invfy1 = mpCurrentKeyFrame->invfy;

    const float linesRatioFactor = 1.5f*mpCurrentKeyFrame->mfScaleFactor;

    int nnewLines=0;
    size_t nlineTotMatches = 0;

    // 算位姿的J矩阵
    Eigen::Matrix3d Rcw_eigen = Converter::toMatrix3d(Rcw1);    //当前帧旋转矩阵3*3
    Eigen::Vector3d tcw_eigen(tcw1.at<float>(0,0), tcw1.at<float>(1,0),tcw1.at<float>(2,0));


    Sophus::SE3 SE3_Rt(Rcw_eigen, tcw_eigen);

    Eigen::Matrix<double, 6, 1> se3 = SE3_Rt.log(); 

    Eigen::Vector3d phi = se3.topRows(3);
    Eigen::Vector3d rho = se3.bottomRows(3);

    Eigen::Matrix<double, 6, 6> J_pose;
    J_pose.setZero();

    J_pose.topLeftCorner(3,3) = hat(phi);
    J_pose.topRightCorner(3,1) = hat(rho);
    J_pose.bottomRightCorner(3,3) = hat(phi);
    J_pose = Eigen::MatrixXd::Identity(6, 6) +  0.5  * J_pose;
    cv::Mat J_pose_f = Converter::toCvMat(J_pose);
    cv::Mat covU = 1.5 * cv::Mat::eye(3,3,CV_32F);
    cv::Mat K1 = mpCurrentKeyFrame->mK.clone();
    cv::Mat K1inv = K1.inv();
    cv::Mat covXc = K1inv * covU * K1inv.t();
    cv::Mat varianceKF1 = mpCurrentKeyFrame->GetDepthVarMat();
    cv::Mat covPose1 = mpCurrentKeyFrame->GetCovpose();
    // 得到伴随矩阵
    cv::Mat Twc1 = Tcw1.inv();
    cv::Mat AdTwc1 = cv::Mat::zeros(6,6,CV_32F);
    cv::Mat Rwc1 = cv::Mat::zeros(3,3,CV_32F);
    cv::Mat twc1 = cv::Mat::zeros(3,1,CV_32F);
    cv::Mat skewtwc1 = cv::Mat::zeros(3,3,CV_32F);
    cv::Mat skewtwcRwc1 = cv::Mat::zeros(3,3,CV_32F);
    Twc1.colRange(0,3).rowRange(0,3).copyTo(Rwc1);
    Twc1.rowRange(0,3).col(3).copyTo(twc1);
    skewtwc1 = skew(twc1);
    skewtwcRwc1 = skewtwc1 * Rwc1;
    Rwc1.copyTo(AdTwc1.rowRange(0,3).colRange(0,3));
    Rwc1.copyTo(AdTwc1.rowRange(3,6).colRange(3,6));
    skewtwcRwc1.copyTo(AdTwc1.rowRange(0,3).colRange(3,6));
    // J_poseinv
    Eigen::Matrix3d Rwc1_eigen = Converter::toMatrix3d(Rwc1);    
    Eigen::Vector3d twc1_eigen = Converter::toVector3d(twc1);  
    Sophus::SE3 SE3_Rt1_inv(Rwc1_eigen,twc1_eigen);
    Eigen::Matrix<double, 6, 1> se31 = SE3_Rt1_inv.log(); 
    Eigen::Vector3d phi1 = se31.topRows(3);
    Eigen::Vector3d rho1 = se31.bottomRows(3);
    Eigen::Matrix<double,6,6> J_poseinv1;
    J_poseinv1.setZero();

    J_poseinv1.topLeftCorner(3,3) = hat(phi1);
    J_poseinv1.topRightCorner(3,1) = hat(rho1);
    J_poseinv1.bottomRightCorner(3,3) = hat(phi1);
    J_poseinv1 = Eigen::MatrixXd::Identity(6, 6) +  0.5  * J_poseinv1;

    cv::Mat J_poseinv1_f = Converter::toCvMat(J_poseinv1);
    const Eigen::Matrix3d K1_eigen = Converter::toMatrix3d(K1);
    // Search matches with epipolar restriction and triangulate
    for(size_t i=0; i<vpNeighKFs.size(); i++)
    {
        if(i>0 && CheckNewKeyFrames())
            return;
        
        KeyFrame* pKF2 = vpNeighKFs[i];

        // Check first that baseline is not too short
        cv::Mat Ow2 = pKF2->GetCameraCenter();
        cv::Mat vBaseline = Ow2-Ow1;
        const float baseline = cv::norm(vBaseline);
        cv::Mat varianceKF2 = pKF2->GetDepthVarMat();
        if(!mbMonocular)
        {
            if(baseline<pKF2->mb)
            continue;
        }
        else
        {
            const float medianDepthKF2 = pKF2->ComputeSceneMedianDepth(2);
            const float ratioBaselineDepth = baseline/medianDepthKF2;

            if(ratioBaselineDepth<0.01)
                continue;
        }

        const cv::Mat &K2 = pKF2->mK;
        cv::Mat K2inv = K2.inv();
        const float minZ = pKF2->mb;
        const float maxZ = 20.f;
        cv::Mat H21,e2;

        computeH21(mpCurrentKeyFrame,pKF2,H21,e2);
        
        std::vector<pair<size_t,size_t>> vMatchedLineIndices;
        pLineMatcher->SearchForTriangulation(mpCurrentKeyFrame,pKF2,vMatchedLineIndices,false);

        cv::Mat Rcw2 = pKF2->GetRotation();
        cv::Mat tcw2 = pKF2->GetTranslation();
        cv::Mat Tcw2 = cv::Mat::eye(4,4,CV_32F);
        Rcw2.copyTo(Tcw2.rowRange(0,3).colRange(0,3));
        tcw2.copyTo(Tcw2.rowRange(0,3).col(3));

        Eigen::Matrix3d Rcw2_eg = Converter::toMatrix3d(Rcw2);
        Eigen::Matrix3d Rwc2_eg = Rcw2_eg.transpose();
        Eigen::Vector3d tcw2_eg = Converter::toVector3d(tcw2);
        Eigen::Vector3d twc2_eg = -Rwc2_eg*tcw2_eg;
        Eigen::Vector3d Ow2_eg = Converter::toVector3d(Ow2);

        const float &fx2 = pKF2->fx;
        const float &fy2 = pKF2->fy;
        const float &cx2 = pKF2->cx;
        const float &cy2 = pKF2->cy;
        const float &invfx2 = pKF2->invfx;
        const float &invfy2 = pKF2->invfy;
        cv::Mat covPose2 = pKF2->GetCovpose();

        // 得到伴随矩阵
        cv::Mat Twc2 = Tcw2.inv();
        cv::Mat AdTwc2 = cv::Mat::zeros(6,6,CV_32F);
        cv::Mat Rwc2 = cv::Mat::zeros(3,3,CV_32F);
        cv::Mat twc2 = cv::Mat::zeros(3,1,CV_32F);
        cv::Mat skewtwc2 = cv::Mat::zeros(3,3,CV_32F);
        cv::Mat skewtwcRwc2 = cv::Mat::zeros(3,3,CV_32F);
        Twc2.colRange(0,3).rowRange(0,3).copyTo(Rwc2);
        Twc2.rowRange(0,3).col(3).copyTo(twc2);
        skewtwc2 = skew(twc2);
        skewtwcRwc2 = skewtwc2 * Rwc2;
        Rwc2.copyTo(AdTwc2.rowRange(0,3).colRange(0,3));
        Rwc2.copyTo(AdTwc2.rowRange(3,6).colRange(3,6));
        skewtwcRwc2.copyTo(AdTwc2.rowRange(0,3).colRange(3,6));
        // J_poseinv
        Eigen::Matrix3d Rwc2_eigen = Converter::toMatrix3d(Rwc2);  
        Eigen::Vector3d twc2_eigen = Converter::toVector3d(twc2);  
        Sophus::SE3 SE3_Rt2_inv(Rwc2_eigen,twc2_eigen);
        Eigen::Matrix<double, 6, 1> se32 = SE3_Rt2_inv.log();
        Eigen::Vector3d phi2 = se32.topRows(3);
        Eigen::Vector3d rho2 = se32.bottomRows(3);
        Eigen::Matrix<double,6,6> J_poseinv2;
        J_poseinv2.setZero();

        J_poseinv2.topLeftCorner(3,3) = hat(phi2);
        J_poseinv2.topRightCorner(3,3) = hat(rho2);
        J_poseinv2.bottomRightCorner(3,3) = hat(phi2);
        J_poseinv2 = Eigen::MatrixXd::Identity(6, 6) +  0.5  * J_poseinv2;

        cv::Mat J_poseinv2_f = Converter::toCvMat(J_poseinv2);
        // Triangulate each line match
        const size_t nlineMatches = vMatchedLineIndices.size();
        nlineTotMatches += nlineMatches;

        const Eigen::Matrix3d K2_eigen = Converter::toMatrix3d(K2);

        for(size_t ikl=0; ikl<nlineMatches; ikl++)
        {
            bool iftriangulation = false;
            int depthSource = 0;
            const int &idx1 = vMatchedLineIndices[ikl].first;
            const int &idx2 = vMatchedLineIndices[ikl].second;

            const cv::line_descriptor::KeyLine& kl1 = mpCurrentKeyFrame->mvKeyLinesUn[idx1]; 
            const float sigma1 = 2.0f; 
            const bool bStereo1 = (mpCurrentKeyFrame->mvuRightLineStart[idx1]>=0) && (mpCurrentKeyFrame->mvuRightLineEnd[idx1]>=0);
            
            const cv::line_descriptor::KeyLine& kl2 = pKF2->mvKeyLinesUn[idx2];
            const float sigma2 = 2.0f;
            const bool bStereo2 = (pKF2->mvuRightLineStart[idx2]>=0) && (pKF2->mvuRightLineEnd[idx2]>=0);
            

            // 直线参数
            const Eigen::Vector3d p1(kl1.startPointX, kl1.startPointY, 1.0);
            const Eigen::Vector3d q1(kl1.endPointX,   kl1.endPointY,   1.0);
            const Eigen::Vector3d m1 = 0.5*(p1+q1);
            Eigen::Vector3d l1 = p1.cross(q1);
            const float l1Norm = sqrt(Utils::Pow2(l1[0])) + Utils::Pow2(l1[1]);
            l1 = l1/l1Norm; // in this way we have l1 = (nx, ny, -d) with (nx^2 + ny^2) = 1

            const Eigen::Vector3d p2(kl2.startPointX, kl2.startPointY, 1.0);
            const Eigen::Vector3d q2(kl2.endPointX,   kl2.endPointY,   1.0);
            const Eigen::Vector3d m2 = 0.5*(p2+q2);
            Eigen::Vector3d l2 = p2.cross(q2); 
            const float l2Norm = sqrt(Utils::Pow2(l2[0]) + Utils::Pow2(l2[1]) );
            l2  = l2/l2Norm; // in this way we have l2 = (nx, ny, -d) with (nx^2 + ny^2) = 1     

            // Check if we can triangulate, i.e. check if the normals of the two planes corresponding to lines are not parallel
            // normals
            bool bCanTriangulateLines = true;
            Eigen::Vector3d n1 = K1_eigen.transpose()*l1; n1 /= n1.norm();
            Eigen::Vector3d n2 = K2_eigen.transpose()*l2; n2 /= n2.norm();

            Eigen::Vector3d n1w = Rwc1_eg*n1;
            Eigen::Vector3d n2w = Rwc2_eg*n2;
            const float normalsDotProduct= fabs( n1w.dot(n2w) );
            const float sigma = std::max( sigma1, sigma2 );
            const float dotProductThreshold = 0.005 * sigma;// 0.002 // 0.005 //Frame::kLineNormalsDotProdThreshold * sigma; // this is a percentage over unitary modulus ( we modulate threshold by sigma)

            if( fabs( normalsDotProduct - 1.f ) < dotProductThreshold) 
            {
                bCanTriangulateLines = false; // normals are almost parallel => cannot triangulate lines
            }

            // Check parallax between rays backprojecting the middle points 
            const Eigen::Vector3d xm1 = Eigen::Vector3d((m1[0]-cx1)*invfx1,   (m1[1]-cy1)*invfy1,   1.0);
            const Eigen::Vector3d xm2 = Eigen::Vector3d((m2[0]-cx2)*invfx2,   (m2[1]-cy2)*invfy2,   1.0);

            const Eigen::Vector3d xn1 = Eigen::Vector3d((p1[0]-cx1)*invfx1,   (p1[1]-cy1)*invfy1,   1.0);
            const Eigen::Vector3d xn2 = Eigen::Vector3d((p2[0]-cx2)*invfx2,   (p2[1]-cy2)*invfy2,   1.0);

            const Eigen::Vector3d xn3 = Eigen::Vector3d((q1[0]-cx1)*invfx1,   (q1[1]-cy1)*invfy1,   1.0);
            const Eigen::Vector3d xn4 = Eigen::Vector3d((q2[0]-cx2)*invfx2,   (q2[1]-cy2)*invfy2,   1.0);

            const Eigen::Vector3d ray1 = Rwc1_eg*xm1;
            const Eigen::Vector3d ray2 = Rwc2_eg*xm2;

            const float cosParallaxRays = ray1.dot(ray2)/(ray1.norm()*ray2.norm()); 

            float cosParallaxStereo = cosParallaxRays+1; 
            float cosParallaxStereo1 = cosParallaxStereo;
            float cosParallaxStereo2 = cosParallaxStereo;

            if(bStereo1 && idx1>=0)
            {
                const float depthM1 = 0.5* ( mpCurrentKeyFrame->mvDepthLineStart[idx1] + mpCurrentKeyFrame->mvDepthLineEnd[idx1] ); // depth middle point left
                cosParallaxStereo1 = cos(2*atan2(mpCurrentKeyFrame->mb/2,depthM1));
            }
            else if(bStereo2 && idx2>=0)
            {
                const float depthM2 = 0.5* ( pKF2->mvDepthLineStart[idx2] + pKF2->mvDepthLineEnd[idx2] ); // depth middle point right                   
                cosParallaxStereo2 = cos(2*atan2(pKF2->mb/2,depthM2));
            }

            cosParallaxStereo = min(cosParallaxStereo1,cosParallaxStereo2);

            Eigen::Vector3d x3DS, x3DE;
            bool bLineTriangulatedByIntersection = false;

            if( bCanTriangulateLines && (cosParallaxRays<cosParallaxStereo) && (cosParallaxRays>0) && (bStereo1 || bStereo2 || cosParallaxRays<0.9998))
            {
                const Eigen::Vector3d e2_eg = Converter::toVector3d(e2);
                const Eigen::Matrix3d H21_eg = Converter::toMatrix3d(H21);

                const float num = -l2.dot(e2_eg);
                const float den1 = (l2.dot(H21_eg*p1)); // distance point-line in pixels 
                const float den2 = (l2.dot(H21_eg*q1)); // distance point-line in pixels 

                constexpr float denTh = 5;
                if( ( fabs(den1) < denTh ) || (fabs(den2) < denTh) ) continue; 

                const float depthP1 = num/den1;
                const float depthQ1 = num/den2;

                if( (depthP1 >= minZ ) && (depthQ1 >= minZ) && (depthP1 <= maxZ) && (depthQ1 <= maxZ) )
                {
                    const Eigen::Vector3d x3DSc = Eigen::Vector3d( (p1[0]-cx1)*invfx1*depthP1,   (p1[1]-cy1)*invfy1*depthP1,   depthP1 );
                    x3DS = Rwc1_eg*x3DSc + twc1_eg;

                    const Eigen::Vector3d x3DEc = Eigen::Vector3d( (q1[0]-cx1)*invfx1*depthQ1,   (q1[1]-cy1)*invfy1*depthQ1,   depthQ1 );  // depthQ1 * K1inv*q1;  
                    x3DE = Rwc1_eg*x3DEc + twc2_eg;

                    Eigen::Vector3d camRay = x3DSc.normalized();
                    Eigen::Vector3d lineES = x3DSc-x3DEc; 
                    const double lineLength = lineES.norm();
                    if(lineLength >= Frame::skMinLineLength3D)
                    {
                        lineES /= lineLength;
                        const float cosViewAngle = fabs((float)camRay.dot(lineES));
                        if(cosViewAngle<=Frame::kCosViewZAngleMax)
                        {
                            if(!bStereo1) 
                            {
                                // assign depth and (virtual) disparity to left line end points 
                                mpCurrentKeyFrame->mvDepthLineStart[idx1] = depthP1;
                                const double disparity_p1 = mpCurrentKeyFrame->mbf/depthP1;                                
                                mpCurrentKeyFrame->mvuRightLineStart[idx1] =  p1(0) - disparity_p1; 

                                mpCurrentKeyFrame->mvDepthLineEnd[idx1] = depthQ1;
                                const double disparity_q1 = mpCurrentKeyFrame->mbf/depthQ1;                         
                                mpCurrentKeyFrame->mvuRightLineEnd[idx1] = q1(0) - disparity_q1; 
                            }

                            bLineTriangulatedByIntersection = true; 
                            iftriangulation = true;
                        }
                    }
                }
            }

            if(!bLineTriangulatedByIntersection)
            {
                if(bStereo1 && cosParallaxStereo1<cosParallaxStereo2)
                {
                    if(!mpCurrentKeyFrame->UnprojectStereoLine(idx1, x3DS, x3DE)) continue; 
                    else
                    {
                        iftriangulation = false; 
                        depthSource = 1;
                    }
                }
                else if(bStereo2 && cosParallaxStereo2<cosParallaxStereo1)
                {
                    if(!pKF2->UnprojectStereoLine(idx2, x3DS, x3DE)) continue; 
                    else
                    {
                        iftriangulation = false;
                        depthSource = 2;
                    }
                }
                else
                {
                    continue;
                }

                //Check triangulation in front of cameras
                const float sz1 = Rcw1_eg.row(2).dot(x3DS)+tcw1_eg[2];
                if(sz1<=0)
                    continue;
                const float ez1 = Rcw1_eg.row(2).dot(x3DE)+tcw1_eg[2];
                if(ez1<=0)
                    continue;
                const float sz2 = Rcw2_eg.row(2).dot(x3DS)+tcw2_eg[2];
                if(sz2<=0)
                    continue;
                const float ez2 = Rcw2_eg.row(2).dot(x3DE)+tcw2_eg[2];
                if(ez2<=0)
                    continue;

                //Check reprojection error in first keyframe
                const float &sigmaSquare1 = 4.0;

                const float sx1 = Rcw1_eg.row(0).dot(x3DS)+tcw1_eg[0];
                const float sy1 = Rcw1_eg.row(1).dot(x3DS)+tcw1_eg[1];
                const float sinvz1 = 1.0/sz1;

                const float su1 = fx1*sx1*sinvz1+cx1;
                const float sv1 = fy1*sy1*sinvz1+cy1;
                const float dl1s = l1[0] * su1 + l1[1] * sv1 + l1[2]; // distance point-line
                if((dl1s*dl1s)>3.84*sigmaSquare1) continue;

                const float ex1 = Rcw1_eg.row(0).dot(x3DE)+tcw1_eg[0];
                const float ey1 = Rcw1_eg.row(1).dot(x3DE)+tcw1_eg[1];
                const float einvz1 = 1.0/ez1; 

                const float eu1 = fx1*ex1*einvz1+cx1;
                const float ev1 = fy1*ey1*einvz1+cy1;
                const float dl1e = l1[0] * eu1 + l1[1] * ev1 + l1[2]; // distance point-line

                if((dl1e*dl1e)>3.84*sigmaSquare1) continue;  


                //Check reprojection error in second keyframe 
                const float sigmaSquare2 = 4.0;

                const float sx2 = Rcw2_eg.row(0).dot(x3DS)+tcw2_eg[0];
                const float sy2 = Rcw2_eg.row(1).dot(x3DS)+tcw2_eg[1];
                const float sinvz2 = 1.0/sz2;  

                const float su2 = fx2*sx2*sinvz2+cx2;
                const float sv2 = fy2*sy2*sinvz2+cy2;
                const float dl2s = l2[0] * su2 + l2[1] * sv2 + l2[2];
                
                if((dl2s*dl2s)>3.84*sigmaSquare2) continue;

                const float ex2 = Rcw2_eg.row(0).dot(x3DE)+tcw2_eg[0];
                const float ey2 = Rcw2_eg.row(1).dot(x3DE)+tcw2_eg[1];
                const float einvz2 = 1.0/ez2; 

                float eu2 = fx2*ex2*einvz2+cx2;
                float ev2 = fy2*ey2*einvz2+cy2;
                const float dl2e = l2[0] * eu2 + l2[1] * ev2 + l2[2];

                if((dl2e*dl2e)>3.84*sigmaSquare2) continue;
            }

            //Check scale consistency
            Eigen::Vector3d x3DM = 0.5*(x3DS+x3DE);

            Eigen::Vector3d normal1 = x3DM-Ow1_eg;
            float dist1 = normal1.norm();


            Eigen::Vector3d normal2 = x3DM-Ow2_eg;
            float dist2 = normal2.norm();

            if(dist1==0 || dist2==0) continue;

            const float linesRatioDist = dist2/dist1;
            const float linesRatioOctave = 1;

            if(linesRatioDist*linesRatioFactor<linesRatioOctave || linesRatioDist>linesRatioOctave*linesRatioFactor) continue;

            Eigen::Vector3d v = x3DE - x3DS;
            Eigen::Vector3d n = v.cross(x3DS);
            Vector6d Plucker;
            Plucker.segment<3>(3) = v;

            MapLine* pML = new MapLine(Plucker,x3DS,x3DE,mpCurrentKeyFrame,mpMap);

            //线的不确定度
            cv::Mat covXwS = cv::Mat::zeros(3,3,CV_32F);
            cv::Mat covXwE = cv::Mat::zeros(3,3,CV_32F);
            cv::Mat pXc_U = cv::Mat::zeros(3,3,CV_32F);
            cv::Mat pXc_z = cv::Mat::zeros(3,1,CV_32F);
            // line uncertainty
            if(iftriangulation)
            {
                // cov_XwS = H1S.inverse() * H2S.transpose() cov(XcS) H2S * H1S.inverse().transpose() +  H1S.inverse() * H3s.transpose() cov(pose) H3s H1S.inverse().transpose()
                // H1iS
                cv::Mat H1iS = cv::Mat::zeros(3,3,CV_32F);
                cv::Mat R = Rcw1.clone();
                cv::Mat t = tcw1.clone();
                cv::Mat UiS = Converter::toCvMat(p1);
                cv::Mat XcS = cv::Mat::zeros(3,1,CV_32F);
                XcS = K1inv * UiS;
                cv::Mat skewXcS = cv::Mat::zeros(3,3,CV_32F);
                skewXcS = skew(XcS);
                H1iS = -2*R.t()*skewXcS*skewXcS*R;
                // H2iS
                cv::Mat H2iS = cv::Mat::zeros(3,3,CV_32F);
                cv::Mat BiS = cv::Mat::zeros(3,1,CV_32F);
                cv::Mat tempS = cv::Mat::zeros(3,3,CV_32F);
                tempS = XcS.dot(BiS)*cv::Mat::eye(3,3,CV_32F) + XcS*BiS.t() - 2 * BiS*XcS.t();
                H2iS = -2*(R.t() * tempS).t();
                // H3iS
                cv::Mat H3iS = cv::Mat::zeros(6,3,CV_32F);
                cv::Mat H3iSt = cv::Mat::zeros(3,6,CV_32F);
                cv::Mat fuskewBiSI = cv::Mat::zeros(3,6,CV_32F);
                cv::Mat fuskewBiS = -skew(BiS);
                fuskewBiS.rowRange(0, 3).colRange(0, 3).copyTo(fuskewBiSI.rowRange(0, 3).colRange(0, 3));
                cv::Mat I33 = cv::Mat::eye(3,3,CV_32F);
                I33.rowRange(0, 3).colRange(0, 3).copyTo(fuskewBiSI.rowRange(0, 3).colRange(3, 6));

                H3iSt = R.t() * skewXcS * skewXcS * fuskewBiSI * J_pose_f;

                H3iS = -2 * H3iSt.t();

                cv::Mat H1iSinv = H1iS.inv();
                covXwS = cv::Mat::zeros(3,3,CV_32F);
                covXwS = H1iSinv * H2iS.t() * covXc * H2iS * H1iSinv.t() + H1iSinv * H3iS.t() * covPose1 * H3iS * H1iSinv.t();

                // cov_XwE = H1E.inverse() * H2E.transpose() cov(XcE) H2E * H1E.inverse().transpose() +  H1E.inverse() * H3s.transpose() cov(pose) H3s H1E.inverse().transpose()
                // H1iE
                cv::Mat H1iE = cv::Mat::zeros(3,3,CV_32F);
                cv::Mat UiE = Converter::toCvMat(p1);
                cv::Mat XcE = cv::Mat::zeros(3,1,CV_32F);
                XcE = K1inv * UiE;
                cv::Mat skewXcE = cv::Mat::zeros(3,3,CV_32F);
                skewXcE = skew(XcE);
                H1iE = -2*R.t()*skewXcE*skewXcE*R;
                // H2iE
                cv::Mat H2iE = cv::Mat::zeros(3,3,CV_32F);
                cv::Mat BiE = cv::Mat::zeros(3,1,CV_32F);
                cv::Mat tempE = cv::Mat::zeros(3,3,CV_32F);
                tempE = XcE.dot(BiE)*cv::Mat::eye(3,3,CV_32F) + XcE*BiE.t() - 2 * BiE*XcE.t();
                H2iE = -2*(R.t() * tempE).t();
                // H3iE
                cv::Mat H3iE = cv::Mat::zeros(6,3,CV_32F);
                cv::Mat H3iEt = cv::Mat::zeros(3,6,CV_32F);
                cv::Mat fuskewBiEI = cv::Mat::zeros(3,6,CV_32F);
                cv::Mat fuskewBiE = -skew(BiE);
                fuskewBiE.rowRange(0, 3).colRange(0, 3).copyTo(fuskewBiEI.rowRange(0, 3).colRange(0, 3));
                I33.rowRange(0, 3).colRange(0, 3).copyTo(fuskewBiEI.rowRange(0, 3).colRange(3, 6));

                H3iEt = R.t() * skewXcE * skewXcE * fuskewBiEI * J_pose_f;

                H3iE = -2 * H3iEt.t();

                cv::Mat H1iEinv = H1iE.inv();
                covXwE = cv::Mat::zeros(3,3,CV_32F);
                covXwE = H1iEinv * H2iE.t() * covXc * H2iE * H1iEinv.t() + H1iEinv * H3iE.t() * covPose1 * H3iE * H1iEinv.t();
                pML->iftriangulation = true;
                pML->ifFused = false;
            }
            
            else //depth
            {
                pML->iftriangulation = false;
                pML->ifFused = true;
                if(depthSource == 1)
                {
                    cv::Mat trans = mpCurrentKeyFrame->GetRotation();
                    int uS = cvRound(p1(0,0));
                    int vS = cvRound(p1(1,0));
                    int uE = cvRound(q1(0,0));
                    int vE = cvRound(q1(1,0));
                    const float &zSC = mpCurrentKeyFrame->mvDepthLineStart[idx1];
                    const float &zEC = mpCurrentKeyFrame->mvDepthLineEnd[idx1];
                    if(uS>=0 && uS<varianceKF1.cols && vS>=0 && vS<varianceKF1.rows
                    && uE>=0 && uE<varianceKF1.cols && vE>=0 && vE<varianceKF1.rows) {
                        float varz_start = varianceKF1.at<float>(vS,uS);
                        float varz_end = varianceKF1.at<float>(vE,uE);

                        cv::Mat startXc =(cv::Mat_<float>(3,1)<<0,0,0);
                        cv::Mat endXc =(cv::Mat_<float>(3,1)<<0,0,0);
                        cv::Mat startU =(cv::Mat_<float>(3,1)<<0,0,0);
                        cv::Mat endU  =(cv::Mat_<float>(3,1)<<0,0,0);

                        startU.at<float>(0,0)=p1(0,0);
                        startU.at<float>(1,0)=p1(1,0);
                        startU.at<float>(2,0)=p1(2,0);

                        endU.at<float>(0,0)=q1(0,0);
                        endU.at<float>(1,0)=q1(1,0);
                        endU.at<float>(2,0)=q1(2,0);

                        startXc = K1inv * startU;
                        endXc = K1inv * endU;
                        
                        const float &u_start=p1(0,0);
                        const float &v_start=p1(1,0);
                        const float &u_end = q1(0,0);
                        const float &v_end = q1(1,0);

                        //covXc = (pXc_U)(cov(U))(pXc_U).t + (pXc_z)(varz)(pXc_z).t
                        cv::Mat covstartXc = cv::Mat::zeros(3,3,CV_32F);
                        cv::Mat pstartXc_U = cv::Mat::zeros(3,3,CV_32F);
                        pstartXc_U.at<float>(0,0) = zSC*invfx1;
                        pstartXc_U.at<float>(1,1) = zSC*invfy1;
                        pstartXc_U.at<float>(2,2) = 1.0f;
                        cv::Mat pstartXc_z = cv::Mat::zeros(3,1,CV_32F);
                        pstartXc_z.at<float>(0,0) = (u_start - cx1) * invfx1;
                        pstartXc_z.at<float>(1,0) = (v_start - cy1) * invfy1;
                        pstartXc_z.at<float>(2,0) = 1.0;

                        covstartXc = pstartXc_U*covU*pstartXc_U.t() + pstartXc_z * varz_start * pstartXc_z.t();

                        cv::Mat covEndXc = cv::Mat::zeros(3,3,CV_32F);
                        cv::Mat pendXc_U = cv::Mat::zeros(3,3,CV_32F);
                        pendXc_U.at<float>(0,0) = zEC*invfx1;
                        pendXc_U.at<float>(1,1) = zEC*invfy1;
                        cv::Mat pendXc_z = cv::Mat::zeros(3,1,CV_32F);
                        pendXc_z.at<float>(0,0) = (u_end - cx1) * invfx1;
                        pendXc_z.at<float>(1,0) = (v_end - cy1) * invfy1;
                        pendXc_z.at<float>(2,0) = 1.0;

                        covEndXc = pendXc_U*covU*pendXc_U.t() + pendXc_z * varz_end * pendXc_z.t();
                        
                        cv::Mat covPoseInv = cv::Mat::zeros(6,6,CV_32F);
                        covPoseInv = AdTwc1 * covPose1 * AdTwc1.t();
                        cv::Mat pXwS_poseinv = cv::Mat::zeros(3,6,CV_32F);
                        const float &xS = x3DS(0,0);
                        const float &yS = x3DS(1,0);
                        const float &zS = x3DS(2,0);
                        cv::Mat pXwS_poseinv_initial = (cv::Mat_<float>(3,6) <<
                         0,  zS, -yS, 1, 0, 0,
                        -zS,  0,  xS, 0, 1, 0,
                        yS, -xS,  0, 0, 0, 1
                        );
                        pXwS_poseinv = pXwS_poseinv_initial * J_poseinv1_f;

                        covXwS = trans * covstartXc * trans.t() + pXwS_poseinv*covPoseInv*pXwS_poseinv.t();

                        cv::Mat pXwE_poseinv = cv::Mat::zeros(3,6,CV_32F);
                        const float &xE = x3DE(0,0);
                        const float &yE = x3DE(1,0);
                        const float &zE = x3DE(2,0);
                        cv::Mat pXwE_poseinv_initial = (cv::Mat_<float>(3,6) <<
                        0,  zE, -yE, 1, 0, 0,
                        -zE,  0,  xE, 0, 1, 0,
                        yE, -xE,  0, 0, 0, 1
                        );
                        pXwE_poseinv = pXwE_poseinv_initial * J_poseinv1_f;
                        covXwE = trans * covEndXc * trans.t() + pXwE_poseinv*covPoseInv*pXwE_poseinv.t();
                    }
                }
                else if(depthSource == 2)
                {
                    cv::Mat trans = pKF2->GetRotation();
                    int uS = cvRound(p2(0,0));
                    int vS = cvRound(p2(1,0));
                    int uE = cvRound(q2(0,0));
                    int vE = cvRound(q2(1,0));
                    const float &zSC = pKF2->mvDepthLineStart[idx2];
                    const float &zEC = pKF2->mvDepthLineEnd[idx2];
                    if(uS>=0 && uS<varianceKF2.cols && vS>=0 && vS<varianceKF2.rows
                    && uE>=0 && uE<varianceKF2.cols && vE>=0 && vE<varianceKF2.rows) {
                        float varz_start = varianceKF2.at<float>(vS,uS);
                        float varz_end = varianceKF2.at<float>(vE,uE);
                        cv::Mat startXc =(cv::Mat_<float>(3,1)<<0,0,0);
                        cv::Mat endXc =(cv::Mat_<float>(3,1)<<0,0,0);
                        cv::Mat startU =(cv::Mat_<float>(3,1)<<0,0,0);
                        cv::Mat endU  =(cv::Mat_<float>(3,1)<<0,0,0);

                        startU.at<float>(0,0)=p2(0,0);
                        startU.at<float>(1,0)=p2(1,0);
                        startU.at<float>(2,0)=p2(2,0);

                        endU.at<float>(0,0)=q2(0,0);
                        endU.at<float>(1,0)=q2(1,0);
                        endU.at<float>(2,0)=q2(2,0);

                        startXc = K2inv * startU;
                        endXc = K2inv * endU;
                        
                        const float &u_start=p2(0,0);
                        const float &v_start=p2(1,0);
                        const float &u_end = q2(0,0);
                        const float &v_end = q2(1,0);

                        //covXc = (pXc_U)(cov(U))(pXc_U).t + (pXc_z)(varz)(pXc_z).t
                        cv::Mat covstartXc = cv::Mat::zeros(3,3,CV_32F);
                        cv::Mat pstartXc_U = cv::Mat::zeros(3,3,CV_32F);
                        pstartXc_U.at<float>(0,0) = zSC*invfx2;
                        pstartXc_U.at<float>(1,1) = zSC*invfy2;
                        pstartXc_U.at<float>(2,2) = 1.0f;
                        cv::Mat pstartXc_z = cv::Mat::zeros(3,1,CV_32F);
                        pstartXc_z.at<float>(0,0) = (u_start - cx2) * invfx2;
                        pstartXc_z.at<float>(1,0) = (v_start - cy2) * invfy2;
                        pstartXc_z.at<float>(2,0) = 1.0;

                        covstartXc = pstartXc_U*covU*pstartXc_U.t() + pstartXc_z * varz_start * pstartXc_z.t();
                        cv::Mat covEndXc = cv::Mat::zeros(3,3,CV_32F);
                        cv::Mat pendXc_U = cv::Mat::zeros(3,3,CV_32F);
                        pendXc_U.at<float>(0,0) = zEC*invfx2;
                        pendXc_U.at<float>(1,1) = zEC*invfy2;
                        cv::Mat pendXc_z = cv::Mat::zeros(3,1,CV_32F);
                        pendXc_z.at<float>(0,0) = (u_end - cx2) * invfx2;
                        pendXc_z.at<float>(1,0) = (v_end - cy2) * invfy2;
                        pendXc_z.at<float>(2,0) = 1.0;

                        covEndXc = pendXc_U*covU*pendXc_U.t() + pendXc_z * varz_end * pendXc_z.t();
                        cv::Mat covPoseInv = cv::Mat::zeros(6,6,CV_32F);
                        covPoseInv = AdTwc2 * covPose2 * AdTwc2.t();
                        cv::Mat pXwS_poseinv = cv::Mat::zeros(3,6,CV_32F);
                        const float &xS = x3DS(0);
                        const float &yS = x3DS(1);
                        const float &zS = x3DS(2);
                        cv::Mat pXwS_poseinv_initial = (cv::Mat_<float>(3,6) <<
                         0,  zS, -yS, 1, 0, 0,
                        -zS,  0,  xS, 0, 1, 0,
                        yS, -xS,  0, 0, 0, 1
                        );
                        pXwS_poseinv = pXwS_poseinv_initial * J_poseinv2_f;
                        covXwS = trans * covstartXc * trans.t() + pXwS_poseinv*covPoseInv*pXwS_poseinv.t();

                        cv::Mat pXwE_poseinv = cv::Mat::zeros(3,6,CV_32F);
                        const float &xE = x3DE(0,0);
                        const float &yE = x3DE(1,0);
                        const float &zE = x3DE(2,0);
                        cv::Mat pXwE_poseinv_initial = (cv::Mat_<float>(3,6) <<
                        0,  zE, -yE, 1, 0, 0,
                        -zE,  0,  xE, 0, 1, 0,
                        yE, -xE,  0, 0, 0, 1
                        );
                        pXwE_poseinv = pXwE_poseinv_initial * J_poseinv2_f;
                        covXwE = trans * covEndXc * trans.t() + pXwE_poseinv*covPoseInv*pXwE_poseinv.t();
                  }
                }
            }
            //不确定度从端点传递到Pluck
            cv::Mat plinestartX = cv::Mat::zeros(6,3,CV_32F);
            cv::Mat endXw = cv::Mat::zeros(3,1,CV_32F);
            cv::Mat startXw = cv::Mat::zeros(3,1,CV_32F);
            startXw = Converter::toCvMat(x3DS);
            endXw = Converter::toCvMat(x3DE);
            cv::Mat skewXe = skew(endXw);
            skewXe.rowRange(0,3).colRange(0,3).copyTo(plinestartX.rowRange(0,3).colRange(0,3));
            cv::Mat  I33 = cv::Mat::eye(3,3,CV_32F);
            I33.rowRange(0,3).colRange(0,3).copyTo(plinestartX.rowRange(3,6).colRange(0,3));
            plinestartX.rowRange(3, 6).colRange(0, 3) = -1 * plinestartX.rowRange(3, 6).colRange(0, 3);
            cv::Mat plineendX = cv::Mat::zeros(6,3,CV_32F);
            cv::Mat skewXs = skew(startXw);
            skewXs.rowRange(0, 3).colRange(0, 3).copyTo(plineendX.rowRange(0, 3).colRange(0, 3));
            plineendX.rowRange(0, 3).colRange(0, 3) = -1 * plineendX.rowRange(0, 3).colRange(0, 3);
            I33.rowRange(0, 3).colRange(0, 3).copyTo(plineendX.rowRange(3, 6).colRange(0, 3));

            cv::Mat var_line = cv::Mat::zeros(6,6,CV_32F);
            var_line = plinestartX * covXwS * plinestartX.t() + plineendX * covXwE * plineendX.t();
            
            pML->SetcovlinePlk(var_line);

            Vector4d orth = plk_to_orth(Plucker);
            pML->SetWorldOR(orth);
            cv::Mat plk(6,1,CV_32F);
            plk = Converter::toCvMat(Plucker);
            cv::Mat pOR_Plk(4,6,CV_32F);
            pOR_Plk = jacobianFromPlktoOrth(plk);

            cv::Mat var_line_or(4,4,CV_32F);
            var_line_or = pOR_Plk * var_line * pOR_Plk.t();

            pML->SetcovlineOR(var_line_or);

            pML->AddObservation(mpCurrentKeyFrame,idx1);
            mpCurrentKeyFrame->AddMapLine(pML,idx1);

            pML->AddObservation(pKF2,idx2);
            pKF2->AddMapLine(pML,idx2);

            pML->ComputeDistinctiveDescriptors();

            pML->UpdateNormalAndDepth();

            mpMap->AddMapLine(pML);
            mlpRecentAddedMapLines.push_back(pML);

            nnewLines++;
        }
    }
}

void LocalMapping::CreateNewMapFeaturesWithUncertainty()
{
    // Retrieve neighbor keyframes in covisibility graph
    int nn = 10;
    if(mbMonocular)
        nn=20;
    const vector<KeyFrame*> vpNeighKFs = mpCurrentKeyFrame->GetBestCovisibilityKeyFrames(nn);
    std::thread threadPoints(&LocalMapping::CreateNewMapPointsUncertainty,this,ref(vpNeighKFs));  
    std::thread threadLines(&LocalMapping::CreateNewMapLinesUncertainty,this,ref(vpNeighKFs));
    threadPoints.join();
    threadLines.join();
}
}