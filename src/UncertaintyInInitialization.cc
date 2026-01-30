#include "UncertaintyInInitialization.h"
#include <thread>
namespace PLUE_SLAM
{   
    cv::Mat skew(const cv::Mat &vec)
    {
        cv::Mat skewM(3,3,CV_32F);
        float vx = vec.at<float>(0,0);
        float vy = vec.at<float>(1,0);
        float vz = vec.at<float>(2,0);
        skewM.at<float>(0,0) = 0.0; skewM.at<float>(0,1) = -vz; skewM.at<float>(0,2) =  vy;
        skewM.at<float>(1,0) = vz;  skewM.at<float>(1,1) = 0.0; skewM.at<float>(1,2) = -vx;
        skewM.at<float>(2,0) = -vy, skewM.at<float>(2,1) = vx;  skewM.at<float>(2,2) = 0.0;
        return skewM.clone();
    }

    Vector4d plk_to_orth(const Vector6d &plk)
    {
        Vector4d orth = Vector4d::Zero();
        Eigen::Vector3d n = Eigen::Vector3d::Zero();
        Eigen::Vector3d v = Eigen::Vector3d::Zero();
        n = plk.topRows(3);
        v = plk.bottomRows(3);
        Eigen::Vector3d u1,u2,u3;
        double w1 = n.norm();
        double w2 = v.norm();
        u1 = n.normalized();
        u2 = v.normalized();
        u3 = u1.cross(v);
        orth(0,0) = atan2(u2(2,0),u3(2,0));
        orth(1,0) = atan2(-u1(2,0),sqrt(u2(2,0)*u2(2,0) + u3(2,0)*u3(2,0)));
        orth(2,0) = atan2(u1(1,0),u1(0,0));
        orth(3,0) = atan2(w2,w1);
        return orth;
    }

    cv::Mat jacobianFromPlktoOrth(const cv::Mat &plk)
    {
        cv::Mat n(3,1,CV_32F);
        cv::Mat v(3,1,CV_32F);
        plk.rowRange(0,3).copyTo(n);
        plk.rowRange(3,6).copyTo(v);
        cv::Mat c(3,1,CV_32F);
        c = n.cross(v);
        const float& nx = plk.at<float>(0,0);
        const float& ny = plk.at<float>(1,0);
        const float& nz = plk.at<float>(2,0);
        const float& vx = plk.at<float>(3,0);
        const float& vy = plk.at<float>(4,0);
        const float& vz = plk.at<float>(5,0);
        const float& cx = c.at<float>(0,0);
        const float& cy = c.at<float>(1,0);
        const float& cz = c.at<float>(2,0);
        const float &Nn = sqrt(nx*nx + ny*ny + nz*nz);
        const float &Nv = sqrt(vx*vx + vy*vy + vz*vz);
        const float &Nc = sqrt(cx*cx + cy*cy + cz*cz);
        cv::Mat pcx_Lj = (cv::Mat_<float>(1,6)<<0,vz,-vy,0,-nz,ny);
        cv::Mat pcy_Lj = (cv::Mat_<float>(1,6)<<-vz,0,vx,nz,0,-nx);
        cv::Mat pcz_Lj = (cv::Mat_<float>(1,6)<<vy,-vx,0,-ny,nz,0);
        // line 1:
        cv::Mat ptheta1_Lj(1,6,CV_32F);
        const float X1 = cz/Nc;
        const float Y1 = vz/Nv;
        const float D1 = X1*X1 + Y1*Y1;
        const float invD1 = 1/D1;
        cv::Mat pY1_Lj(1,6,CV_32F);
        const float invNv3 = 1/(Nv*Nv*Nv);
        pY1_Lj = (cv::Mat_<float>(1,6)<<0.f,0.f,0.f,(-vx*vy)*invNv3,(-vy*vz)*invNv3,(Nv*Nv-vz*vz)*invNv3);
        cv::Mat pX1_Lj(1,6,CV_32F);
        // pX1_Lj = (Nc pcz_Lj - cz pNc_Lj) / (Nc*Nc);
        cv::Mat pNc_Lj(1,6,CV_32F);
        const float invNc = 1/Nc;
        const float invNc2 = invNc * invNc;
        pNc_Lj = invNc * (cx * pcx_Lj + cy * pcy_Lj + cz * pcz_Lj); 
        pX1_Lj = (Nc * pcz_Lj - cz * pNc_Lj) * invNc2;
        ptheta1_Lj = (X1*pY1_Lj - Y1*pX1_Lj) * invD1;
        // line 2:
        cv::Mat ptheta2_Lj(1,6,CV_32F);
        const float Y2 = -nz/Nn;
        const float X2 = sqrt(D1);
        const float D2 = X2*X2+Y2*Y2;
        const float invD2 = 1/D2;
        cv::Mat pY2_Lj(1,6,CV_32F);
        const float invNn3 = 1/(Nn*Nn*Nn);
        pY2_Lj = (cv::Mat_<float>(1,6)<<(nx*nz)*invNn3,(ny*nz)*invNn3,(nz*nz-Nn*Nn)*invNn3,0.f,0.f,0.f);
        cv::Mat pX2_Lj(1,6,CV_32F);
        const float invX2 = 1/X2;
        pX2_Lj = (X1*pX1_Lj + Y1*pY1_Lj) * invX2;
        ptheta2_Lj = (X2*pY2_Lj - Y2*pX2_Lj) * invD2;
        // line 3:
        cv::Mat ptheta3_Lj(1,6,CV_32F);
        const float Y3 = ny/Nn;
        const float X3 = nx/Nn;
        const float D3 = X3*X3 + Y3*Y3;
        const float invD3 = 1/D3;
        cv::Mat pY3_Lj(1,6,CV_32F);
        pY3_Lj = (cv::Mat_<float>(1,6)<<(-nx*ny)*invNn3,(Nn*Nn-ny*ny)*invNn3,(-ny*nz)*invNn3,0.f,0.f,0.f);
        cv::Mat pX3_Lj(1,6,CV_32F);
        pX3_Lj = (cv::Mat_<float>(1,6)<<(Nn*Nn-nx*nx)*invNn3,(-nx*ny)*invNn3,(-nx*nz)*invNn3,0.f,0.f,0.f);
        ptheta3_Lj = (X3*pY3_Lj - Y3*pX3_Lj) * invD3;
        // line 4:
        cv::Mat pphi_Lj(1,6,CV_32F);
        const float Y4 = Nv;
        const float X4 = Nn;
        const float D4 = X4*X4+Y4*Y4;
        const float invD4 = 1/D4;
        cv::Mat pY4_Lj(1,6,CV_32F);
        pY4_Lj = (cv::Mat_<float>(1,6)<<0.f,0.f,0.f,vx/Nv,vy/Nv,vz/Nv);
        cv::Mat pX4_Lj(1,6,CV_32F);
        pX4_Lj = (cv::Mat_<float>(1,6)<<nx/Nn,ny/Nn,nz/Nn,0.f,0.f,0.f);
        pphi_Lj = (X4*pY4_Lj - Y4*pX4_Lj) * invD4;

        cv::Mat jac(4,6,CV_32F);
        ptheta1_Lj.copyTo(jac.colRange(0,6).row(0));
        ptheta2_Lj.copyTo(jac.colRange(0,6).row(1));
        ptheta3_Lj.copyTo(jac.colRange(0,6).row(2));
        pphi_Lj.copyTo(jac.colRange(0,6).row(3));
        return jac.clone();
    }

    cv::Mat VD6toCvMat(const Eigen::Matrix<double,6,1> &Vec)
    {
        cv::Mat cvMat(6,1,CV_32F);
        for(int i=0;i<6;i++)
                cvMat.at<float>(i)=Vec(i);
        return cvMat.clone();
    }
    
    void UncertaintyInInitialization::StereoInitializationPointUncertainty(ORB_SLAM2::Tracking &mTracking, cv::Mat& variance, ORB_SLAM2::KeyFrame* pKFini)
    {
        const float fx = mTracking.mCurrentFrame.fx, fy = mTracking.mCurrentFrame.fy, cx = mTracking.mCurrentFrame.cx, cy = mTracking.mCurrentFrame.cy;

        // Create MapPoints and asscoiate to KeyFrame
        for(int i=0; i<mTracking.mCurrentFrame.N;i++)
        {
            float z_ini = mTracking.mCurrentFrame.mvDepth[i];
            if(z_ini > 0)
            {
                const cv::KeyPoint &kp = mTracking.mCurrentFrame.mvKeys[i];
                const float &uu = kp.pt.x;
                const float &vv = kp.pt.y; 
                int u = cvRound(kp.pt.x);
                int v = cvRound(kp.pt.y);
                if(u>=0 && u<variance.cols && v>=0 && v<variance.rows) {
                float z   = z_ini;
                float varz = variance    .at<float>(v, u);
                const cv::KeyPoint &kpUn = mTracking.mCurrentFrame.mvKeysUn[i];
                const float uuUn = kpUn.pt.x;
                const float vvUn = kpUn.pt.y;
                //covXW = (pXw_U)(cov(U))(pXw_U).t + (pXw_z)(varz)(pXw_z).t
                cv::Mat covXw = cv::Mat::zeros(3,3,CV_32F);
                cv::Mat pXw_U = cv::Mat::zeros(3,3,CV_32F);
                pXw_U.at<float>(0,0) = z/fx;
                pXw_U.at<float>(1,1) = z/fy;
                cv::Mat covU = 1.5 * cv::Mat::eye(3,3,CV_32F);
                covU.at<float>(2,2) = 0.0;
                cv::Mat pXw_z = cv::Mat::zeros(3,1,CV_32F);
                pXw_z.at<float>(0,0) = (uuUn - cx) / fx;
                pXw_z.at<float>(1,0) = (vvUn - cy) / fy;
                pXw_z.at<float>(2,0) = 1.0;
                covXw = pXw_U*covU*pXw_U.t() + pXw_z * varz * pXw_z.t();

                cv::Mat var=(cv::Mat_<float>(3,3)<<0,0,0,0,0,0,0,0,0);// 
                var=covXw.clone(); 

                cv::Mat x3D = mTracking.mCurrentFrame.UnprojectStereo(i);
                ORB_SLAM2::MapPoint* pNewMP = new ORB_SLAM2::MapPoint(x3D,pKFini,mTracking.mpMap); // added get Map
                pNewMP->SetCovtri3(var); // added
                pNewMP->iftriangulation=false; // added
                pNewMP->ifFused=true; // added

                pNewMP->AddObservation(pKFini,i);
                pKFini->AddMapPoint(pNewMP,i);
                pNewMP->ComputeDistinctiveDescriptors();
                pNewMP->UpdateNormalAndDepth();
                mTracking.mpMap->AddMapPoint(pNewMP); // added get Map
                mTracking.mCurrentFrame.mvpMapPoints[i]=pNewMP;
            }
            }
        }
    }

    void UncertaintyInInitialization::StereoInitializationLineUncertainty(ORB_SLAM2::Tracking &mTracking, cv::Mat& variance, ORB_SLAM2::KeyFrame* pKFini)
    {
        const float& cx = mTracking.mCurrentFrame.cx;
        const float& cy = mTracking.mCurrentFrame.cy;
        const float& fx = mTracking.mCurrentFrame.fx;
        const float& fy = mTracking.mCurrentFrame.fy;
        for(int i=0; i<mTracking.mCurrentFrame.NL;i++) // NL: added line number
        {
            const float& zS = mTracking.mCurrentFrame.mvDepthLineStart[i]; // added line startPoints
            const float& zE = mTracking.mCurrentFrame.mvDepthLineEnd[i]; // added line endPoints
            if( (zS>0) && (zE>0) )                
            {
                Eigen::Vector3d x3DS, x3DE;
                if(mTracking.mCurrentFrame.UnprojectStereoLine(i,x3DS,x3DE)) // added line construction
                {
                    // TODO: add distance 
                    Eigen::Vector3d v = x3DE - x3DS;
                    v/=v.norm();
                    // TODO: norm or not?
                    Eigen::Vector3d n = x3DS.cross(x3DE);
                    n/=n.norm();
                    Vector6d Plucker;
                    Plucker.segment<3>(0) = n;
                    Plucker.segment<3>(3) = v;
                    float u_start = mTracking.mCurrentFrame.mvKeyLinesUn[i].startPointX; // mvKeyLinesUn: added line observations
                    float v_start = mTracking.mCurrentFrame.mvKeyLinesUn[i].startPointY;
                    float u_end = mTracking.mCurrentFrame.mvKeyLinesUn[i].endPointX;
                    float v_end = mTracking.mCurrentFrame.mvKeyLinesUn[i].endPointY;
                    if(u_start>=0 && u_start<variance.cols && v_start>=0 && v_start<variance.rows
                    && u_end>=0 && u_end<variance.cols && v_end>=0 && v_end<variance.rows) {
                        float varz_start = variance.at<float>(v_start, u_start);
                        float varz_end = variance.at<float>(v_end, u_end);
                        cv::Mat startXw =(cv::Mat_<float>(3,1)<<0,0,0);
                        cv::Mat endXw =(cv::Mat_<float>(3,1)<<0,0,0);
                        startXw.at<float>(0,0)=x3DS(0,0);
                        startXw.at<float>(1,0)=x3DS(1,0);
                        startXw.at<float>(2,0)=x3DS(2,0);

                        endXw.at<float>(0,0)=x3DE(0,0);
                        endXw.at<float>(1,0)=x3DE(1,0);
                        endXw.at<float>(2,0)=x3DE(2,0);

                        //covXW = (pXw_U)(cov(U))(pXw_U).t + (pXw_z)(varz)(pXw_z).t
                        cv::Mat covXwS=  cv::Mat::zeros(3,3,CV_32F);
                        cv::Mat pstartXw_U = cv::Mat::zeros(3,3,CV_32F);
                        pstartXw_U.at<float>(0,0) = zS/fx;
                        pstartXw_U.at<float>(1,1) = zS/fy;
                        pstartXw_U.at<float>(2,2) = 0.0f;
                        cv::Mat covU = 1.5 * cv::Mat::eye(3,3,CV_32F); // 100 *
                        cv::Mat pstartXw_z = cv::Mat::zeros(3,1,CV_32F);
                        pstartXw_z.at<float>(0,0) = (u_start - cx) / fx;
                        pstartXw_z.at<float>(1,0) = (v_start - cy) / fy;
                        pstartXw_z.at<float>(2,0) = 1.0;
                        covXwS = pstartXw_U*covU*pstartXw_U.t() + pstartXw_z * varz_start * pstartXw_z.t();
                        cv::Mat covXwE = cv::Mat::zeros(3,3,CV_32F);
                        cv::Mat pendXw_U = cv::Mat::zeros(3,3,CV_32F);
                        pendXw_U.at<float>(0,0) = zE/fx;
                        pendXw_U.at<float>(1,1) = zE/fy;
                        cv::Mat pendXw_z = cv::Mat::zeros(3,1,CV_32F);
                        pendXw_z.at<float>(0,0) = (u_end - cx) / fx;
                        pendXw_z.at<float>(1,0) = (v_end - cy) / fy;
                        pendXw_z.at<float>(2,0) = 0.0;
                        covXwE = pendXw_U*covU*pendXw_U.t() + pendXw_z * varz_end * pendXw_z.t();
                        cv::Mat plinestartX = cv::Mat::zeros(6,3,CV_32F);
                        cv::Mat skewXe = skew(endXw);
                        skewXe.rowRange(0, 3).colRange(0, 3).copyTo(plinestartX.rowRange(0, 3).colRange(0, 3));
                        cv::Mat I33 = cv::Mat::eye(3,3,CV_32F);
                        I33.rowRange(0, 3).colRange(0, 3).copyTo(plinestartX.rowRange(3, 6).colRange(0, 3));
                        plinestartX.rowRange(3, 6).colRange(0, 3) = -1 * plinestartX.rowRange(3, 6).colRange(0, 3);
                        cv::Mat plineendX = cv::Mat::zeros(6,3,CV_32F);
                        cv::Mat skewXs = skew(startXw);
                        skewXs.rowRange(0, 3).colRange(0, 3).copyTo(plineendX.rowRange(0, 3).colRange(0, 3));
                        plineendX.rowRange(0, 3).colRange(0, 3) = -1 * plineendX.rowRange(0, 3).colRange(0, 3);
                        I33.rowRange(0, 3).colRange(0, 3).copyTo(plineendX.rowRange(3, 6).colRange(0, 3));

                        cv::Mat var_line = cv::Mat::zeros(6,6,CV_32F);
                        var_line = plinestartX * covXwS * plinestartX.t() + plineendX * covXwE * plineendX.t();

                        ORB_SLAM2::MapLine* pNewLine = new ORB_SLAM2::MapLine(Plucker,x3DS,x3DE,pKFini,mTracking.mpMap); // added get map
                        pNewLine->SetcovlinePlk(var_line);
                        Vector4d orth = plk_to_orth(Plucker);
                        pNewLine->SetWorldOR(orth);
                        cv::Mat plk(6,1,CV_32F);
                        plk = VD6toCvMat(Plucker); //added Vector6d->cv::Mat
                        cv::Mat pOR_Plk(4,6,CV_32F);
                        pOR_Plk = jacobianFromPlktoOrth(plk);
                        cv::Mat var_line_or(4,4,CV_32F);
                        var_line_or = pOR_Plk * var_line * pOR_Plk.t();
                        pNewLine->ifFused=true;
                        pNewLine->SetcovlineOR(var_line_or);
                        pNewLine->AddObservation(pKFini,i);
                        pKFini->AddMapLine(pNewLine,i); // added Add MapLine
                        pNewLine->ComputeDistinctiveDescriptors();
                        pNewLine->UpdateNormalAndDepth();
                        mTracking.mpMap->AddMapLine(pNewLine);
                        mTracking.mCurrentFrame.mvpMapLines[i]=pNewLine; // added MapLines
                    }
                }
            }   
        }
    }

    void UncertaintyInInitialization::StereoInitializationPointLineWithUncertainty(ORB_SLAM2::Tracking &mTracking)
    {
        if(mTracking.mCurrentFrame.N > 500)
        {
            // Set Frame pose to the origin
            mTracking.mCurrentFrame.SetPose(cv::Mat::eye(4,4,CV_32F));
            mTracking.mCurrentFrame.covpose = cv::Mat::eye(6,6,CV_32F) * 0.01; //added covpose
            // Create KeyFrame
            ORB_SLAM2::KeyFrame* pKFini = new ORB_SLAM2::KeyFrame(mTracking.mCurrentFrame,mTracking.mpMap,mTracking.mpKeyFrameDB); //added: get map and get keyframeDB

            const cv::Mat depth = mTracking.mCurrentFrame.imDepth.clone(); // added depth image
            cv::Mat depth_smoothed;
            cv::Mat variance;
            cv::Mat gausskernel = (cv::Mat_<float>(3, 3) << 1.0f, 2.0f, 1.0f, 2.0f, 4.0f, 2.0f, 1.0f, 2.0f, 1.0f);
            const float c = 16.0f;

            // d_sensor_uncertainty = 0.0000285*z^2  
            cv::filter2D(depth, depth_smoothed, CV_32F, gausskernel, cv::Point(-1,-1), 0.0, cv::BORDER_REPLICATE);
            depth_smoothed /= c;
            cv::Mat depth2;
            cv::multiply(depth, depth, depth2);

            cv::Mat var_in = depth2 * 0.0000285f;

            cv::Mat combined = depth2 + var_in;

            cv::Mat mean_combined;
            cv::filter2D(combined, mean_combined, CV_32F, gausskernel, cv::Point(-1,-1), 0.0, cv::BORDER_REPLICATE);
            mean_combined /= c;

            // 4) Var = E[depth^2 + σ_in^2] - (E[depth])^2
            variance = mean_combined - depth_smoothed.mul(depth_smoothed);
            mTracking.mCurrentFrame.setDepthVarMat(variance);
            pKFini->SetDepthVarMat(variance); // added depth var
            std::thread threadPoints(&UncertaintyInInitialization::StereoInitializationPointUncertainty,this,pKFini,ref(variance),ref(mTracking.mCurrentFrame),mTracking.mpMap);  
            std::thread threadLines(&UncertaintyInInitialization::StereoInitializationLineUncertainty,this,pKFini,ref(variance),ref(mTracking.mCurrentFrame),mTracking.mpMap);
            threadPoints.join();
            threadLines.join();

            cout << "New map created with " << mTracking.mpMap->MapPointsInMap() << " points and " << mTracking.mpMap->MapLinesInMap() << " lines." << endl;

            // ORB SLAM2 process
            mTracking.mpLocalMapper->InsertKeyFrame(pKFini);

            mTracking.mLastFrame = ORB_SLAM2::Frame(mCurrentFrame);
            mTracking.mnLastKeyFrameId=mCurrentFrame.mnId;
            mTracking.mpLastKeyFrame = pKFini;

            mTracking.mvpLocalKeyFrames.push_back(pKFini);
            mTracking.mvpLocalMapPoints=mpMap->GetAllMapPoints();
            mTracking.mvpLocalMapLines=mpMap->GetAllMapLines();
            mTracking.mpReferenceKF = pKFini;
            mCurrentFrame.mpReferenceKF = pKFini;

            mpMap->SetReferenceMapPoints(mTracking.mvpLocalMapPoints);
            mpMap->SetReferenceMapLines(mTracking.mvpLocalMapLines);

            mpMap->mvpKeyFrameOrigins.push_back(pKFini);

            mTracking.mpMapDrawer->SetCurrentCameraPose(mCurrentFrame.mTcw);
            mTracking.mState=OK;
        }
    }
}