#include "UncertaintyInTracking.h"
#include <random>
#include <sophus/se3.h>


namespace PLUE_SLAM
{   
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
    
    Eigen::Matrix3d hat(const Eigen::Vector3d &v)
    {
        Eigen::Matrix3d Omega;
        Omega <<  0, -v(2),  v(1)
            ,  v(2),     0, -v(0)
            , -v(1),  v(0),     0;
        return Omega;
    }

    cv::Mat Mat6DtoCvMat(const Eigen::Matrix<double,6,6> &m)
    {
        cv::Mat cvMat(6,6,CV_32F);
        for(int i=0;i<6;i++)
            for(int j=0; j<6; j++)
                cvMat.at<float>(i,j)=m(i,j);

        return cvMat.clone();
    }

    cv::Mat transformationLines(const cv::Mat &T)
    {
        cv::Mat transLine = cv::Mat::zeros(6,6,CV_32F);
        cv::Mat Rcw(3,3,CV_32F);
        Rcw = T.rowRange(0,3).colRange(0,3);
        cv::Mat tcw(3,1,CV_32F);
        tcw = T.rowRange(0,3).col(3);
        Rcw.copyTo(transLine.rowRange(0,3).colRange(0,3));
        cv::Mat skewtcwRcw(3,3,CV_32F);
        skewtcwRcw = skew(tcw)*Rcw;
        skewtcwRcw.copyTo(transLine.rowRange(0,3).colRange(3,6));
        Rcw.copyTo(transLine.rowRange(3,6).colRange(3,6));
        return transLine.clone();
    }

    void UncertaintyInTracking::ComputePointMatrixH4(cv::Mat &J_pose, cv::Mat &H4, ORB_SLAM2::Tracking &mTracking)
    {
        // H4 = H41 + H42 
        const float &fx = mTracking.mCurrentFrame.fx;
        const float &fy = mTracking.mCurrentFrame.fy;
        const float &cx = mTracking.mCurrentFrame.cx;
        const float &cy = mTracking.mCurrentFrame.cy;
        cv::Mat H41 = cv::Mat::zeros(6,6,CV_32F);
        cv::Mat H42 = cv::Mat::zeros(6,6,CV_32F);
        int countMp = 0;
        for(int i = 0; i < mTracking.mCurrentFrame.N; ++i)
        {
            ORB_SLAM2::MapPoint *pMP = mTracking.mCurrentFrame.mvpMapPoints[i];
            if(pMP)
            {
                if (pMP->Observations() < 1)           
                {
                    mTracking.mCurrentFrame.mvbOutlier[i] = false;
                    mTracking.mCurrentFrame.mvpMapPoints[i] = static_cast<ORB_SLAM2::MapPoint *>(NULL);
                }
                else
                {
                    countMp++;
                    cv::Mat H41i = cv::Mat::zeros(6,6,CV_32F); // H41_i = 2*(pei_pose).t()(pei_pose)
                    cv::Mat H42i = cv::Mat::zeros(6,6,CV_32F); // H42_i = 2*ei.t() * ppei_pose_pose
                    cv::Mat pi = cv::Mat::zeros(2,1,CV_32F);
                    const cv::KeyPoint &kp = mTracking.mCurrentFrame.mvKeysUn[i];
                    const float &ui = kp.pt.x;
                    const float &vi = kp.pt.y;
                    pi = (cv::Mat_<float>(2,1) << ui,vi);
                    cv::Mat Xw = pMP->GetWorldPos();
                    cv::Mat Xw4 = (cv::Mat_<float>(4, 1) << Xw.at<float>(0, 0), Xw.at<float>(1, 0), Xw.at<float>(2, 0), 1.0f); 
                    cv::Mat Xc4 = mTracking.mCurrentFrame.mTcw * Xw4;
                    const float &x = Xc4.at<float>(0,0);
                    const float &y = Xc4.at<float>(1,0);
                    const float &z = Xc4.at<float>(2,0);
                    const float &invz = 1.0  / Xc4.at<float>(0,2);
                    const float &invz_2 = invz * invz;
                    cv::Mat ei = (cv::Mat_<float>(2, 1) << 0, 0);     
                    cv::Mat proj_i = (cv::Mat_<float>(2,1) << (fx * x) * invz + cx, (fy * y) * invz + cy);
                    ei = pi - proj_i;
                    // H41_i = 2 * (pei_pose).t()(pei_pose)
                    cv::Mat pei_pose = cv::Mat::zeros(2,6,CV_32F);
                    // pei_pose = pei_Xc * pXc_pose
                    cv::Mat pei_Xc = (cv::Mat_<float>(2,3) <<
                    -fx*invz, 0,        fx*x*invz_2,
                    0,        -fy*invz, fy*y*invz_2
                    );
                    cv::Mat pXc_pose_initial = (cv::Mat_<float>(3,6) <<
                    0,  z, -y, 1, 0, 0,
                    -z,  0,  x, 0, 1, 0,
                    y, -x,  0, 0, 0, 1
                    );

                    cv::Mat pXc_pose = cv::Mat::zeros(3,6,CV_32F);
                    pXc_pose = pXc_pose_initial * J_pose;

                    pei_pose = (cv::Mat_<float>(2,6) <<
                    x*y*invz_2 *fx,    -(1+(x*x*invz_2)) *fx,  y*invz *fx,  -invz *fx,         0, x*invz_2 *fx,
                    (1+y*y*invz_2) *fy,      -x*y*invz_2 *fy, -x*invz *fy,          0, -invz *fy, y*invz_2 *fy);

                    pei_pose = pei_pose * J_pose;
                    H41i = 2 * pei_pose.t() * pei_pose;

                    // H42_i =2 * ei.t() * ppei_pose_pose
                    cv::Mat ppei0_Xc_Xc  = (cv::Mat_<float>(3,3)  << 0,         0,         fx*invz_2,
                                                                    0,         0,         0,
                                                                    fx*invz_2, 0,         -2*fx*x*invz*invz_2);
                    cv::Mat ppei1_Xc_Xc =  (cv::Mat_<float>(3,3)  << 0,         0,         0,
                                                                    0,         0,         fy*invz_2,
                                                                    0,         fy*invz_2, -2*fy*y*invz*invz_2);
                    cv::Mat H421_i = cv::Mat::zeros(6,6,CV_32F);
                    const float &ei0 = ei.at<float>(0,0);
                    const float &ei1 = ei.at<float>(1,0);
                    H421_i = 2 *  (pXc_pose).t() * (ei0 * ppei0_Xc_Xc + ei1 * ppei1_Xc_Xc)  * pXc_pose;

                    cv::Mat H422_i = cv::Mat::zeros(6,6,CV_32F);
                    cv::Mat H422_i1 = cv::Mat::zeros(6,6,CV_32F);
                    cv::Mat H422_i2 = cv::Mat::zeros(6,6,CV_32F);
                    // H422_i = H422i1 + H422i2
                    cv::Mat eit_pei_Xc = cv::Mat::zeros(1,3,CV_32F);
                    eit_pei_Xc = 2 * ei.t() * pei_Xc;
                    const float& t1 = eit_pei_Xc.at<float>(0,0);
                    const float& t2 = eit_pei_Xc.at<float>(0,1);
                    const float& t3 = eit_pei_Xc.at<float>(0,2);
                    cv::Mat H422_i1_1 = cv::Mat::zeros(6,6,CV_32F);
                    cv::Mat tem1 = (cv::Mat_<float>(6,3) << 0,0,0,0,0,1,0,-1,0,0,0,0,0,0,0,0,0,0);
                    H422_i1_1 = J_pose.t() * tem1 * pXc_pose;
                    cv::Mat H422_i1_2 = cv::Mat::zeros(6,6,CV_32F);
                    cv::Mat tem2 = (cv::Mat_<float>(6,3) << 0,0,-1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0);
                    H422_i1_2 = J_pose.t() * tem2 * pXc_pose;
                    cv::Mat H422_i1_3 = cv::Mat::zeros(6,6,CV_32F);
                    cv::Mat tem3 = (cv::Mat_<float>(6,3) << 0,1,0,-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0);
                    H422_i1_3 = J_pose.t() * tem3 * pXc_pose;
                    H422_i1 = t1 * H422_i1_1 + t2 * H422_i1_2 + t3 * H422_i1_3;

                    cv::Mat temb = cv::Mat::zeros(1,6,CV_32F);
                    temb = eit_pei_Xc * pXc_pose_initial;

                    const float& b1 = temb.at<float>(0,0);
                    const float& b2 = temb.at<float>(0,1);
                    const float& b3 = temb.at<float>(0,2);
                    const float& b4 = temb.at<float>(0,3);
                    const float& b5 = temb.at<float>(0,4);
                    const float& b6 = temb.at<float>(0,5);

                    cv::Mat H422_i2_1 = cv::Mat::zeros(6,6,CV_32F);
                    H422_i2_1.at<float>(1,2) = 0.5;
                    H422_i2_1.at<float>(2,1) = -0.5;
                    H422_i2_1.at<float>(4,5) = 0.5;
                    H422_i2_1.at<float>(5,4) = -0.5;
                    cv::Mat H422_i2_2 = cv::Mat::zeros(6,6,CV_32F);
                    H422_i2_2.at<float>(0,2) = -0.5;
                    H422_i2_2.at<float>(2,0) = 0.5;
                    H422_i2_2.at<float>(3,5) = -0.5;
                    H422_i2_2.at<float>(5,3) = 0.5;
                    cv::Mat H422_i2_3 = cv::Mat::zeros(6,6,CV_32F);
                    H422_i2_3.at<float>(0,1) = 0.5;
                    H422_i2_3.at<float>(1,0) = -0.5;
                    H422_i2_3.at<float>(3,4) = 0.5;
                    H422_i2_3.at<float>(4,3) = -0.5;
                    cv::Mat H422_i2_4 = cv::Mat::zeros(6,6,CV_32F);
                    H422_i2_4.at<float>(1,5) = 0.5;
                    H422_i2_4.at<float>(2,4) = -0.5;
                    cv::Mat H422_i2_5 = cv::Mat::zeros(6,6,CV_32F);
                    H422_i2_5.at<float>(0,5) = -0.5;
                    H422_i2_5.at<float>(2,3) = 0.5;
                    cv::Mat H422_i2_6 = cv::Mat::zeros(6,6,CV_32F);
                    H422_i2_6.at<float>(1,3) = -0.5;
                    H422_i2_6.at<float>(0,4) = 0.5;
                    H422_i2 = b1 * H422_i2_1 + b2 * H422_i2_2 + b3 * H422_i2_3 + b4 * H422_i2_4 + b5 * H422_i2_5 + b6 * H422_i2_6;
                    H422_i = H422_i1 + H422_i2;   
                    H42i = H421_i + H422_i;
                    H41 += H41i;
                    H42 += H42i;
                }
            }
        }
        H4 = H41+H42;
    }

    void UncertaintyInTracking::ComputeLineMatrixH7(cv::Mat &Jl, cv::Mat &H7, ORB_SLAM2::Tracking &mTracking)
    {
        const float &fx = mTracking.mCurrentFrame.fx;
        const float &fy = mTracking.mCurrentFrame.fy;
        const float &cx = mTracking.mCurrentFrame.cx;
        const float &cy = mTracking.mCurrentFrame.cy;
        cv::Mat Rcw(3,3,CV_32F);
        Rcw = mTracking.mCurrentFrame.mTcw.rowRange(0,3).colRange(0,3);
        cv::Mat tcw(3,1,CV_32F);
        tcw = mTracking.mCurrentFrame.mTcw.rowRange(0,3).col(3);
        cv::Mat Kl = (cv::Mat_<float>(3,3) << fy,0,0,0,fx,0,-fy*cx,-fx*cy,fx*fy);
        cv::Mat H71 = cv::Mat::zeros(6,6,CV_32F); // (55) term1
        cv::Mat H72 = cv::Mat::zeros(6,6,CV_32F); // (55) term2
        int countLines = 0;
        for (int j = 0; j < mTracking.mCurrentFrame.NL; j++) // added line number
        {                                                 
            ORB_SLAM2::MapLine *pML = mTracking.mCurrentFrame.mvpMapLines[j]; // added MapLines.
            if (pML && (pML->Observations() > 1))
            {
                cv::Mat H71j = cv::Mat::zeros(6,6,CV_32F); // H71j = 2 * (pej_pose).t() * (pej_pose)
                cv::Mat H72j = cv::Mat::zeros(6,6,CV_32F); // H72j = 2 * ej.t() * ppej_pose_pose

                Eigen::Matrix<double,6,1> Lw_eigen = pML->GetWorldPlk();
                Eigen::Vector3d nw_eigen = Lw_eigen.head(3);
                Eigen::Vector3d vw_eigen = Lw_eigen.tail(3);
                cv::Mat nw = ORB_SLAM2::Converter::toCvMat(nw_eigen);
                cv::Mat vw = ORB_SLAM2::Converter::toCvMat(vw_eigen);
                cv::Mat tcw_skew = skew(tcw);
                cv::Mat nc(3,1,CV_32F);
                nc = Rcw*nw + tcw_skew*Rcw*vw;
                cv::Mat vc(3,1,CV_32F);
                vc = Rcw*vw;

                cv::Mat lineProj(3,1,CV_32F);
                lineProj = Kl * nc;

                const float &l1 = lineProj.at<float>(0,0);
                const float &l2 = lineProj.at<float>(1,0);
                const float &l3 = lineProj.at<float>(2,0);

                const float &xs1 = mTracking.mCurrentFrame.mvKeyLinesUn[j].startPointX; //added line observation
                const float &xs2 = mTracking.mCurrentFrame.mvKeyLinesUn[j].startPointY;
                const float &xe1 = mTracking.mCurrentFrame.mvKeyLinesUn[j].endPointX;
                const float &xe2 = mTracking.mCurrentFrame.mvKeyLinesUn[j].endPointY;
                
                const float &deno2 = sqrt(l1*l1+l2*l2);
                const float &deno1 = deno2 * deno2 * deno2;
                const float &invdeno1 = 1.0  / deno1;
                const float &invdeno2 = 1.0  / deno2;    
                float ejs = (xs1 * l1 + xs2 * l2 + l3) * invdeno2;
                float eje = (xe1 * l1 + xe2 * l2 + l3) * invdeno2;
                if(ejs*ejs >= 5.991 || eje*eje >= 5.991)
                {
                    continue;
                }
                const float &pej_lj00 = (xs1*l2*l2 - l1*l2*xs2 - l1*l3) * invdeno1;
                const float &pej_lj01 = (xs2*l1*l1 - l1*l2*xs1 - l2*l3) * invdeno1;
                const float &pej_lj02 = 1 * invdeno2;
                const float &pej_lj10 = (xe1*l2*l2 - l1*l2*xe2 - l1*l3) * invdeno1;
                const float &pej_lj11 = (xe2*l1*l1 - l1*l2*xe1 - l2*l3) * invdeno1;
                const float &pej_lj12 = 1 * invdeno2;

                // pej_pose = pej_lj * plj_Lj * pLj_pose
                cv::Mat pej_pose = cv::Mat::zeros(2,6,CV_32F);
                cv::Mat pej_lj = cv::Mat::zeros(2,3,CV_32F); // 2*3 (56)
                pej_lj = (cv::Mat_<float>(2,3) << pej_lj00, pej_lj01, pej_lj02,
                                                pej_lj10, pej_lj11, pej_lj12);

                cv::Mat plj_Lj = cv::Mat::zeros(3,6,CV_32F); // 3*6 (57)
                Kl.copyTo(plj_Lj.rowRange(0,3).colRange(0,3));
                
                cv::Mat pLj_pose = cv::Mat::zeros(6,6,CV_32F); // 6*6 (67)
                cv::Mat minusnc_skewJl(3,3,CV_32F); 
                minusnc_skewJl = -skew(nc) * Jl;
                cv::Mat minusvc_skewJl(3,3,CV_32F);
                minusvc_skewJl = -skew(vc);
                minusnc_skewJl.copyTo(pLj_pose.rowRange(0,3).colRange(0,3));
                minusvc_skewJl.copyTo(pLj_pose.rowRange(0,3).colRange(3,6));
                minusvc_skewJl.copyTo(pLj_pose.rowRange(3,6).colRange(0,3));
                pej_pose = pej_lj * plj_Lj * pLj_pose;
                H71j = 2 * pej_pose.t() * pej_pose;
                // H72j = 2 * ej.t() * ppej_pose_pose 
                // H72j = H721_j + H722_j + H723_j 
                // H721j = plj_pose.t() * (ej0 * p(pej_lj(0))_lj.t() + ej1 * p(ej_lj(1))_lj.t()) * plj_pose
                // pej_lj0_lj = a1 ,a2, a3    pej_lj1_lj = a4, a5, a6 
                //              a2, b2, b3                 a5, b5, b6
                //              a3, b3, c3                 a6, b6, c6
                cv::Mat ej = (cv::Mat_<float>(2,1)<<(xs1*l1+xs2*l2+l3)*invdeno2,(xe1*l1+xe2*l2+l3)*invdeno2);
                const float &ej0 = ej.at<float>(0,0);
                const float &ej1 = ej.at<float>(1,0);
                const float &deno3 = deno2 * deno2 * deno2 * deno2 *deno2;
                const float &invdeno3 = 1.0  / deno3;
                const float &a1 = (-3*l1*l2*l2*xs1+2*l1*l1*l2*xs2-l2*l2*l2*xs2+2*l1*l1*l3-l2*l2*l3)*invdeno3;
                const float &a2 = (2*l1*l1*l2*xs1+2*l1*l2*l2*xs2-l2*l2*l2*xs1-l1*l1*l1*xs2+3*l1*l2*l3)*invdeno3;
                const float &a3 = -l1 * invdeno1;
                const float &a4 = (-3*l1*l2*l2*xe1+2*l1*l1*l2*xe2-l2*l2*l2*xe2+2*l1*l1*l3-l2*l2*l3)*invdeno3;
                const float &a5 = (2*l1*l1*l2*xe1+2*l1*l2*l2*xe2-l2*l2*l2*xe1-l1*l1*l1*xe2+3*l1*l2*l3)*invdeno3;
                const float &a6 = -l1 * invdeno1;
                const float &b2 = (2*l1*l2*l2*xs1-3*l1*l1*l2*xs2-l1*l1*l1*xs1+2*l2*l2*l3-l1*l1*l3)*invdeno3;
                const float &b3 = -l2*invdeno1;
                const float &b5 = (2*l1*l2*l2*xe1-3*l1*l1*l2*xe2-l1*l1*l1*xe1+2*l2*l2*l3-l1*l1*l3)*invdeno3;
                const float &b6 = -l2*invdeno1;
                const float &c3 = 0.0f;
                const float &c6 = 0.0f;
                cv::Mat ppej_lj0_lj = (cv::Mat_<float>(3,3)<< a1,a2,a3,a2,b2,b3,a3,b3,c3);
                cv::Mat ppej_lj1_lj = (cv::Mat_<float>(3,3)<< a4,a5,a6,a5,b5,b6,a6,b6,c6);
                cv::Mat H721j = cv::Mat::zeros(6,6,CV_32F);
                cv::Mat plj_pose(3,6,CV_32F);
                plj_pose = plj_Lj * pLj_pose;
                H721j = plj_pose.t() * (ej0 * ppej_lj0_lj + ej1 * ppej_lj1_lj) * plj_pose;
                // H722j = 0;
                // H723j = 2ej.t() * pej_lj * plj_Lj * ppLj_pose_pose
                // ppLj_pose_pose = ((ppLj_pose_Lj).t() * pLj_pose).t()
                // tem723 = 2ej.t() * pej_lj * plj_Lj  1*6
                cv::Mat tem723(1,6,CV_32F);
                tem723 = 2*ej.t() * pej_lj * plj_Lj; // 
                const float &t1 = tem723.at<float>(0,0);
                const float &t2 = tem723.at<float>(0,1);
                const float &t3 = tem723.at<float>(0,2);
                // H723j = t1*(pLj_pose).t()*(p(pLj_pose(0))_Lj).t() +t2*(pLj_pose).t()*(p(pLj_pose(1))_Lj).t()+t3*(pLj_pose).t()*p(pLj_pose(2))_Lj).t()
                cv::Mat ppLj_pose0_Lj = cv::Mat::zeros(6,6,CV_32F);
                cv::Mat ppLj_pose1_Lj = cv::Mat::zeros(6,6,CV_32F);
                cv::Mat ppLj_pose2_Lj = cv::Mat::zeros(6,6,CV_32F);
                cv::Mat one1 = (cv::Mat_<float>(3,1)<<1.0f, 0.0f, 0.0f);
                cv::Mat one2 = (cv::Mat_<float>(3,1)<<0.0f, 1.0f, 0.0f);
                cv::Mat one3 = (cv::Mat_<float>(3,1)<<0.0f, 0.0f, 1.0f);
                cv::Mat skewone1Jl(3,3,CV_32F);
                cv::Mat skewone2Jl(3,3,CV_32F);
                cv::Mat skewone3Jl(3,3,CV_32F);
                skewone1Jl = skew(one1) * Jl;
                skewone2Jl = skew(one2) * Jl;
                skewone3Jl = skew(one3);
                skewone1Jl.copyTo(ppLj_pose0_Lj.rowRange(0,3).colRange(0,3)); 
                skewone1Jl.copyTo(ppLj_pose0_Lj.rowRange(3,6).colRange(3,6));
                skewone2Jl.copyTo(ppLj_pose1_Lj.rowRange(0,3).colRange(0,3));
                skewone2Jl.copyTo(ppLj_pose1_Lj.rowRange(3,6).colRange(3,6)); 
                skewone3Jl.copyTo(ppLj_pose2_Lj.rowRange(0,3).colRange(0,3));
                skewone3Jl.copyTo(ppLj_pose2_Lj.rowRange(3,6).colRange(3,6));
                cv::Mat H723j = cv::Mat::zeros(6,6,CV_32F);
                H723j = t1*(pLj_pose).t()*ppLj_pose0_Lj +t2*(pLj_pose).t()*ppLj_pose1_Lj+t3*(pLj_pose).t()*ppLj_pose2_Lj;
                H72j = H721j+H723j;
                H71 += H71j;
                H72 += H72j;
            }
        }
        H7 = H71+H72;
    }

    void UncertaintyInTracking::ComputeUncertaintyFromPoints(cv::Mat &J_pose, cv::Mat &H4, cv::Mat &H7, cv::Mat &pose_uncertainty_point, ORB_SLAM2::Tracking &mTracking)
    {
        const float &fx = mTracking.mCurrentFrame.fx;
        const float &fy = mTracking.mCurrentFrame.fy;
        const float &cx = mTracking.mCurrentFrame.cx;
        const float &cy = mTracking.mCurrentFrame.cy;
        cv::Mat covU = 1.5 * cv::Mat::eye(2,2,CV_32F);
        cv::Mat Rcw(3,3,CV_32F);
        Rcw = mTracking.mCurrentFrame.mTcw.rowRange(0,3).colRange(0,3);
        cv::Mat tcw(3,1,CV_32F);
        tcw = mTracking.mCurrentFrame.mTcw.rowRange(0,3).col(3);
        cv::Mat dominv = cv::Mat::zeros(6,6,CV_32F), dom = cv::Mat::zeros(6,6,CV_32F);
        dominv = (H4+H7);
        dom = dominv.inv();
        cv::Mat tem_pose_uncertainty_point(6,6,CV_32F);
        tem_pose_uncertainty_point = cv::Mat::zeros(6,6,CV_32F);
        for(int i = 0; i < mTracking.mCurrentFrame.N; ++i)
        {
            ORB_SLAM2::MapPoint *pMP = mTracking.mCurrentFrame.mvpMapPoints[i];
            if(pMP)
            {
                if (pMP->Observations() < 1)           
                {
                    mTracking.mCurrentFrame.mvbOutlier[i] = false;
                    mTracking.mCurrentFrame.mvpMapPoints[i] = static_cast<ORB_SLAM2::MapPoint *>(NULL);
                }
                else
                {
                    cv::Mat pi = cv::Mat::zeros(2,1,CV_32F);
                    const cv::KeyPoint &kp = mTracking.mCurrentFrame.mvKeysUn[i];
                    const float &ui = kp.pt.x;
                    const float &vi = kp.pt.y;
                    pi = (cv::Mat_<float>(2,1) << ui,vi);
                    cv::Mat Xw = pMP->GetWorldPos();
                    cv::Mat Xw4 = (cv::Mat_<float>(4, 1) << Xw.at<float>(0, 0), Xw.at<float>(1, 0), Xw.at<float>(2, 0), 1.0f); 
                    cv::Mat Xc4 = mTracking.mCurrentFrame.mTcw * Xw4;
                    const float &x = Xc4.at<float>(0,0);
                    const float &y = Xc4.at<float>(1,0);
                    const float &z = Xc4.at<float>(2,0);
                    const float &invz = 1.0  / Xc4.at<float>(2,0);
                    const float &invz_2 = invz * invz;
                    cv::Mat ei = (cv::Mat_<float>(2, 1) << 0, 0);     
                    cv::Mat proj_i = (cv::Mat_<float>(2,1) << (fx * x) * invz + cx, (fy * y) * invz + cy);
                    ei = pi - proj_i;
                    // H5_i = 2 * pei_pose
                    cv::Mat H5_i(2,6,CV_32F);
                    cv::Mat pei_pose = (cv::Mat_<float>(2,6) <<
                    x*y*invz_2 *fx,    -(1+(x*x*invz_2)) *fx,  y*invz *fx,  -invz *fx,         0, x*invz_2 *fx,
                    (1+y*y*invz_2) *fy,      -x*y*invz_2 *fy, -x*invz *fy,          0, -invz *fy, y*invz_2 *fy);
                    pei_pose = pei_pose * J_pose;
                    H5_i = 2 * pei_pose;
                    // H6_i = H61_i + H62_i 
                    // H61_i = 2*(pei_Xw).t() * pei_pose pei_Xw = pei_Xc * pXc_Xw = pei_Xc * Rcw;
                    cv::Mat pei_Xc = (cv::Mat_<float>(2,3) <<
                    -fx*invz, 0,        fx*x*invz_2,
                    0,        -fy*invz, fy*y*invz_2
                    );
                    cv::Mat pei_Xw(2,3,CV_32F);
                    pei_Xw = pei_Xc * Rcw;
                    cv::Mat H61_i(3,6,CV_32F);
                    H61_i = 2 * pei_Xw.t() * pei_pose;
                    // H62_i = 2*ei.t() * ppei_pose_Xw = 2*(ei(0)*Rcw.t()*(ppei_pose0_Xc).t()+ei(1)*Rcw.t()*(ppei_pose1_Xc).t();
                    const float &ei0 = ei.at<float>(0,0);
                    const float &ei1 = ei.at<float>(1,0);
                    cv::Mat ppei_pose0_Xc(3,6,CV_32F);
                    cv::Mat ppei_pose1_Xc(3,6,CV_32F);
                    const float &invz_3 = invz * invz_2;
                    ppei_pose0_Xc = (cv::Mat_<float>(3,6) << 
                        y*invz_2*fx,  -2*x*invz_2*fx,            0,         0, 0,      fx*invz_2,
                        x*invz_2*fx,               0,      fx*invz,         0, 0,              0,
                    -2*x*y*invz_3 * fx, 2*x*x*invz_3*fx, -y*invz_2*fx, fx*invz_2, 0, -2*x*invz_3*fx);

                    ppei_pose1_Xc = (cv::Mat_<float>(3,6) <<
                                0,    -y*invz_2*fy,     -invz*fy, 0,         0,              0,
                    2*y*invz_2*fy,    -x*invz_2*fy,            0, 0,         0,      invz_2*fy,
                    -2*y*y*invz_3, 2*x*y*invz_3*fy, -x*invz_2*fy, 0, invz_2*fy, -2*y*invz_3*fy);
                    cv::Mat H62_i(3,6,CV_32F);
                    H62_i = 2*(ei0*Rcw.t()*(ppei_pose0_Xc)+ei1*Rcw.t()*(ppei_pose1_Xc));
                    cv::Mat H6_i(3,6,CV_32F);
                    H6_i = H61_i + H62_i;
                    cv::Mat covXw = pMP->GetCovtri3(); // add get uncertainty
                    tem_pose_uncertainty_point += H5_i.t() * covU * H5_i + H6_i.t() * covXw * H6_i; 
                }
            }
        }
        pose_uncertainty_point = dom * tem_pose_uncertainty_point * dom.t();
    }

    void UncertaintyInTracking::ComputeUncertaintyFromLines(cv::Mat &Jl, cv::Mat &H4, cv::Mat &H7, cv::Mat &pose_uncertainty_line, ORB_SLAM2::Tracking &mTracking)
    {
        const float &fx = mTracking.mCurrentFrame.fx;
        const float &fy = mTracking.mCurrentFrame.fy;
        const float &cx = mTracking.mCurrentFrame.cx;
        const float &cy = mTracking.mCurrentFrame.cy;
        cv::Mat covU = 1.5 * cv::Mat::eye(2,2,CV_32F);
        cv::Mat Rcw(3,3,CV_32F);
        Rcw = mTracking.mCurrentFrame.mTcw.rowRange(0,3).colRange(0,3);
        cv::Mat tcw(3,1,CV_32F);
        tcw = mTracking.mCurrentFrame.mTcw.rowRange(0,3).col(3);
        cv::Mat Kl = (cv::Mat_<float>(3,3) << fy,0,0,0,fx,0,-fy*cx,-fx*cy,fx*fy);
        cv::Mat dominv = cv::Mat::zeros(6,6,CV_32F), dom = cv::Mat::zeros(6,6,CV_32F);
        dominv = (H4+H7);
        dom = dominv.inv();
        cv::Mat tem_pose_uncertainty_line(6,6,CV_32F);
        tem_pose_uncertainty_line = cv::Mat::zeros(6,6,CV_32F);
        for( int j = 0; j < mTracking.mCurrentFrame.NL; j++ ) // added line number
        {
            ORB_SLAM2::MapLine *pML = mTracking.mCurrentFrame.mvpMapLines[j]; // added mapLines
            
            if (pML && (pML->Observations() >= 1))
            {
                Eigen::Matrix<double,6,1> Lw_eigen = pML->GetWorldPlk();
                Eigen::Vector3d nw_eigen = Lw_eigen.head(3);
                Eigen::Vector3d vw_eigen = Lw_eigen.tail(3);
                cv::Mat nw = ORB_SLAM2::Converter::toCvMat(nw_eigen);
                cv::Mat vw = ORB_SLAM2::Converter::toCvMat(vw_eigen);
                cv::Mat Ljw (6,1,CV_32F);
                nw.copyTo(Ljw.rowRange(0,3).col(0));
                vw.copyTo(Ljw.rowRange(0,3).col(0));
                cv::Mat tcw_skew = skew(tcw);

                cv::Mat nc(3,1,CV_32F);
                nc = Rcw*nw + tcw_skew*Rcw*vw;
                cv::Mat vc(3,1,CV_32F);
                vc = Rcw*vw;

                cv::Mat lineProj(3,1,CV_32F);
                lineProj = Kl * nc;
                const float &l1 = lineProj.at<float>(0,0);
                const float &l2 = lineProj.at<float>(1,0);
                const float &l3 = lineProj.at<float>(2,0);

                const float &xs1 = mTracking.mCurrentFrame.mvKeyLinesUn[j].startPointX; // added line observation
                const float &xs2 = mTracking.mCurrentFrame.mvKeyLinesUn[j].startPointY;
                const float &xe1 = mTracking.mCurrentFrame.mvKeyLinesUn[j].endPointX;
                const float &xe2 = mTracking.mCurrentFrame.mvKeyLinesUn[j].endPointY;

                const float &deno2 = sqrt(l1*l1+l2*l2);
                const float &deno1 = deno2 * deno2 *deno2;
                const float &invdeno1 = 1.0  / deno1;
                const float &invdeno2 = 1.0  / deno2;    
                float ejs = (xs1 * l1 + xs2 * l2 + l3) * invdeno2;
                float eje = (xe1 * l1 + xe2 * l2 + l3) * invdeno2;
                if(ejs*ejs >= 5.991 || eje*eje >= 5.991)
                {
                    continue;
                }
                const float &pej_lj00 = (xs1*l2*l2 - l1*l2*xs2 - l1*l3) * invdeno1;
                const float &pej_lj01 = (xs2*l1*l1 - l1*l2*xs1 - l2*l3) * invdeno1;
                const float &pej_lj02 = 1 * invdeno2;
                const float &pej_lj10 = (xe1*l2*l2 - l1*l2*xe2 - l1*l3) * invdeno1;
                const float &pej_lj11 = (xe2*l1*l1 - l1*l2*xe1 - l2*l3) * invdeno1;
                const float &pej_lj12 = 1 * invdeno2;

                // pej_pose = pej_lj * plj_Lj * pLj_pose
                cv::Mat pej_pose(2,6,CV_32F); // 2*6
                cv::Mat pej_lj = cv::Mat::zeros(2,3,CV_32F); // 2*3 (56)
                pej_lj = (cv::Mat_<float>(2,3) << pej_lj00, pej_lj01, pej_lj02,
                                                pej_lj10, pej_lj11, pej_lj12);

                cv::Mat plj_Lj = cv::Mat::zeros(3,6,CV_32F); // 3*6 (57)
                Kl.copyTo(plj_Lj.rowRange(0,3).colRange(0,3));

                cv::Mat pLj_pose = cv::Mat::zeros(6,6,CV_32F); // 6*6 (67)
                cv::Mat minusnc_skewJl(3,3,CV_32F); 
                minusnc_skewJl = -skew(nc) * Jl;
                cv::Mat minusvc_skewJl(3,3,CV_32F);
                minusvc_skewJl = -skew(vc);
                minusnc_skewJl.copyTo(pLj_pose.rowRange(0,3).colRange(0,3));
                minusvc_skewJl.copyTo(pLj_pose.rowRange(0,3).colRange(3,6));
                minusvc_skewJl.copyTo(pLj_pose.rowRange(3,6).colRange(0,3));

                pej_pose = pej_lj * plj_Lj * pLj_pose;
                // H8_j = 2*(pej_xsj).t() * pej_pose + 2 * ej.t() * ppej_pose_xsj = H81_j + H82_j
                cv::Mat H81_j(2,6,CV_32F);
                cv::Mat pej_xsj = (cv::Mat_<float>(2,2)<< l1*invdeno2, l2*invdeno2, 0, 0);
                H81_j = 2 * pej_xsj.t() * pej_pose;
                // H82_j = 2*ej.t()*ppej_pose_xsj
                // ppej_pose_xsj = ppej_lj_xsj * plj_Lj * pLj_pose
                // H82_j = 2*(ej(0)*ppej_lj0_xsj + ej(1) * ppej_lj1_xsj) * plj_Lj * pLj_pose
                // ppej_lj1_xsj = 0;
                cv::Mat ej = (cv::Mat_<float>(2,1)<<(xs1*l1+xs2*l2+l3)*invdeno2,(xe1*l1+xe2*l2+l3)*invdeno2);
                const float &ej0 = ej.at<float>(0,0);
                const float &ej1 = ej.at<float>(1,0);
                cv::Mat H82_j(2,6,CV_32F);
                cv::Mat ppej_lj0_xsj = (cv::Mat_<float>(2,3)<< l2*l2*invdeno1, -l1*l2*invdeno1, 0, -l1*l2*invdeno1, l1*l1*invdeno1, 0);
                H82_j = 2 * ej0 * ppej_lj0_xsj * plj_Lj * pLj_pose;
                cv::Mat H8_j(2,6,CV_32F);
                H8_j = H81_j + H82_j;

                // H9_j = 2*(pej_xej).t() * pej_pose + 2 * ej.t() * ppej_pose_xej = H91_j + H92_j
                cv::Mat H91_j(2,6,CV_32F);
                cv::Mat pej_xej = (cv::Mat_<float>(2,2)<< 0, 0, l1*invdeno2, l2*invdeno2);
                H91_j = 2 * pej_xej.t() * pej_pose;
                // H92_j = 2*ej.t()*ppej_pose_xej
                // ppej_pose_xej = ppej_lj_xej * plj_Lj * pLj_pose
                // H82_j = 2*(ej(0)*ppej_lj0_xej + ej(1) * ppej_lj1_xej) * plj_Lj * pLj_pose
                // ppej_lj0_xsj = 0;
                cv::Mat H92_j(2,6,CV_32F);
                cv::Mat ppej_lj1_xej = (cv::Mat_<float>(2,3)<< l2*l2*invdeno1, -l1*l2*invdeno1, 0, -l1*l2*invdeno1, l1*l1*invdeno1, 0);
                H92_j = 2 * ej1 * ppej_lj1_xej * plj_Lj * pLj_pose;
                cv::Mat H9_j(2,6,CV_32F);
                H9_j = H91_j + H92_j;

                //H10_j = 2 * pej_Ljw.t()*pej_pose + 2 * ej.t()*ppej_pose_Ljw = H101_j + H102_j
                //pej_Ljw = pej_lj * plj_Lj * pLj_Ljw * pLjw_Oj
                cv::Mat transLine = cv::Mat::zeros(6,6,CV_32F);
                transLine = transformationLines(mTracking.mCurrentFrame.mTcw);
                cv::Mat pLj_Ljw(6,6,CV_32F); // 6*6
                pLj_Ljw = transLine.clone();
                cv::Mat pej_Ljw(2,6,CV_32F); // 2*6
                pej_Ljw = pej_lj * plj_Lj * pLj_Ljw;
                cv::Mat H101_j(6,6,CV_32F); // 6*6
                H101_j = 2 * pej_Ljw.t() * pej_pose;


                const float &deno3 = deno2 * deno2 * deno2 * deno2 *deno2;
                const float &invdeno3 = 1.0  / deno3;
                const float &a1 = (-3*l1*l2*l2*xs1+2*l1*l1*l2*xs2-l2*l2*l2*xs2+2*l1*l1*l3-l2*l2*l3)*invdeno3;
                const float &a2 = (2*l1*l1*l2*xs1+2*l1*l2*l2*xs2-l2*l2*l2*xs1-l1*l1*l1*xs2+3*l1*l2*l3)*invdeno3;
                const float &a3 = -l1 * invdeno1;
                const float &a4 = (-3*l1*l2*l2*xe1+2*l1*l1*l2*xe2-l2*l2*l2*xe2+2*l1*l1*l3-l2*l2*l3)*invdeno3;
                const float &a5 = (2*l1*l1*l2*xe1+2*l1*l2*l2*xe2-l2*l2*l2*xe1-l1*l1*l1*xe2+3*l1*l2*l3)*invdeno3;
                const float &a6 = -l1 * invdeno1;
                const float &b2 = (2*l1*l2*l2*xs1-3*l1*l1*l2*xs2-l1*l1*l1*xs1+2*l2*l2*l3-l1*l1*l3)*invdeno3;
                const float &b3 = -l2*invdeno1;
                const float &b5 = (2*l1*l2*l2*xe1-3*l1*l1*l2*xe2-l1*l1*l1*xe1+2*l2*l2*l3-l1*l1*l3)*invdeno3;
                const float &b6 = -l2*invdeno1;
                const float &c3 = 0.0f;
                const float &c6 = 0.0f;
                // H102_j =  2 * ej.t()*ppej_pose_Oj
                // ppej_pose = ppej_lj_Oj * plj_Lj * pLj_pose + pej_lj * pplj_Lj_Oj * pLj_pose + pej_lj * plj_Lj + ppLj_pose_Oj
                // H102_j = H1021_j + H1022_j + H1023_j
                // H1021_j = 2 * ej.t()* ppej_lj_Oj * plj_Lj * pLj_pose
                // 2*ej.t()*ppej_lj_Oj = 2*plj_Oj.t() * (ej(0)*ppej_lj0_lj.t() + ej(1)*ppej_lj1_lj.t())
                cv::Mat ppej_lj0_lj = (cv::Mat_<float>(3,3)<< a1,a2,a3,a2,b2,b3,a3,b3,c3);
                cv::Mat ppej_lj1_lj = (cv::Mat_<float>(3,3)<< a4,a5,a6,a5,b5,b6,a6,b6,c6);
                cv::Mat H1021_j(6,6,CV_32F);
                cv::Mat plj_Ljw(3,6,CV_32F);
                plj_Ljw = plj_Lj * pLj_Ljw ;
                H1021_j = 2 * plj_Ljw.t() * (ej0 * ppej_lj0_lj + ej1 * ppej_lj1_lj) * plj_Lj * pLj_pose;
                // H1022_j = 2 * ej.t() * pej_lj * pplj_Lj_Oj * pLj_pose = 0
                // H1023_j = 2 * ej.t() * pej_lj * plj_Lj * ppLj_pose_Oj = 2 * tem *ppLj_pose_Oj
                // H1023_j = 2 * pLj_Oj.t() * (tem(0) * ppLj_pose0_Lj + tem(1) * ppLj_pose1_Lj + tem(2) * ppLj_pose2_Lj)
                cv::Mat tem(1,6,CV_32F);
                tem = ej.t() * pej_lj * plj_Lj;
                const float &t0 = tem.at<float>(0,0);
                const float &t1 = tem.at<float>(0,1);
                const float &t2 = tem.at<float>(0,2);
                cv::Mat ppLj_pose0_Lj = cv::Mat::zeros(6,6,CV_32F);
                cv::Mat ppLj_pose1_Lj = cv::Mat::zeros(6,6,CV_32F);
                cv::Mat ppLj_pose2_Lj = cv::Mat::zeros(6,6,CV_32F);
                cv::Mat one1 = (cv::Mat_<float>(3,1)<<1.0f, 0.0f, 0.0f);
                cv::Mat one2 = (cv::Mat_<float>(3,1)<<0.0f, 1.0f, 0.0f);
                cv::Mat one3 = (cv::Mat_<float>(3,1)<<0.0f, 0.0f, 1.0f);
                cv::Mat skewone1Jl(3,3,CV_32F);
                cv::Mat skewone2Jl(3,3,CV_32F);
                cv::Mat skewone3Jl(3,3,CV_32F);
                skewone1Jl = skew(one1) * Jl;
                skewone2Jl = skew(one2) * Jl;
                skewone3Jl = skew(one3);
                skewone1Jl.copyTo(ppLj_pose0_Lj.rowRange(0,3).colRange(0,3)); 
                skewone1Jl.copyTo(ppLj_pose0_Lj.rowRange(3,6).colRange(3,6));
                skewone2Jl.copyTo(ppLj_pose1_Lj.rowRange(0,3).colRange(0,3));
                skewone2Jl.copyTo(ppLj_pose1_Lj.rowRange(3,6).colRange(3,6)); 
                skewone3Jl.copyTo(ppLj_pose2_Lj.rowRange(0,3).colRange(0,3));
                skewone3Jl.copyTo(ppLj_pose2_Lj.rowRange(3,6).colRange(3,6));
                // H1023_j = 2 * pLj_Oj.t() * (tem(0) * ppLj_pose0_Lj + tem(1) * ppLj_pose1_Lj + tem(2) * ppLj_pose2_Lj)
                cv::Mat H1023_j(6,6,CV_32F);
                H1023_j = 2 * pLj_Ljw.t() * (t0 * ppLj_pose0_Lj + t1 * ppLj_pose1_Lj + t2 * ppLj_pose2_Lj);
                cv::Mat H102_j(6,6,CV_32F); // 6*6
                H102_j = H1021_j + H1023_j;
                cv::Mat H10_j(6,6,CV_32F);
                H10_j = H101_j + H102_j;
                cv::Mat covLwPlk = pML->GetcovlinePlk();
                tem_pose_uncertainty_line += H8_j.t() * covU * H8_j + H9_j.t() * covU * H9_j + H10_j.t() * covLwPlk * H10_j;
            }
        }
        pose_uncertainty_line = dom * tem_pose_uncertainty_line * dom.t();
    }

    // functions for CI-Filter
    struct PSOParams {
    int swarm_size = 30;
    int max_iters  = 100;
    double w_inertia = 0.7;
    double c1 = 1.5;
    double c2 = 1.5;
    };

    void CIFilter(const cv::Mat &x1, const cv::Mat &x2, const cv::Mat &P1, const cv::Mat &P2, float w, cv::Mat &xFused, cv::Mat &PFused)
    {
        int dim = x1.rows;
        cv::Mat I1 = P1.inv();
        cv::Mat I2 = P2.inv();

        cv::Mat I(dim,dim,CV_32F);

        I = w*I1 + (1.0-w)*I2;

        PFused = I.inv();

        cv::Mat info(3,3,CV_32F);
        info = w * I1 * x1 + (1.0 - w) * I2 * x2;
        xFused = PFused * info;
    }

    float matTrace(const cv::Mat &M)
    {
        return cv::trace(M)[0];
    }

    float optimizeWbyPSO(const cv::Mat &x1, const cv::Mat &x2, const cv::Mat &P1, const cv::Mat &P2, const PSOParams &params)
    {
        const int dim = x1.rows;
        std::mt19937 rng(27);
        std::uniform_real_distribution<float> uni(0.0, 1.0);

        struct Particle {
            float pos, vel;
            float pbest_pos, pbest_val;
        };

        // initialization
        std::vector<Particle> swarm;
        swarm.reserve(params.swarm_size);
        for (int i = 0; i < params.swarm_size; i++) {
            float w0 = uni(rng);
            swarm.push_back({w0, 0.0, w0, std::numeric_limits<float>::infinity()});
        }

        // best swarm
        float gbest_pos = 0.0;
        float gbest_val = std::numeric_limits<float>::infinity();

        for (auto& p : swarm) {
            cv::Mat xf(dim,1,CV_32F); 
            cv::Mat Pf(dim,dim,CV_32F);
            CIFilter(x1, x2, P1, P2, p.pos, xf, Pf);
            float val = matTrace(Pf);
            p.pbest_val = val;
            if (val < gbest_val) {
                gbest_val = val;
                gbest_pos = p.pos;
            }
        }

        for (int iter = 0; iter < params.max_iters; iter++) {
            for (auto& p : swarm) {
                float r1 = uni(rng), r2 = uni(rng);
                p.vel = params.w_inertia * p.vel
                        + params.c1 * r1 * (p.pbest_pos - p.pos)
                        + params.c2 * r2 * (gbest_pos - p.pos);
                p.pos = std::min((float)1.0, std::max((float)0.0, p.pos + p.vel));

                cv::Mat xf(dim,1,CV_32F);
                cv::Mat Pf(dim,dim,CV_32F);
                CIFilter(x1, x2, P1, P2, p.pos, xf, Pf);
                float val = matTrace(Pf);

                if (val < p.pbest_val) {
                    p.pbest_val = val;
                    p.pbest_pos = p.pos;
                }

                if (val < gbest_val) {
                    gbest_val = val;
                    gbest_pos = p.pos;
                }
            }
        }
        return gbest_pos;
    }

    struct pointTobeFused
    {
        int pointIndex = -1;
        cv::Mat pointDepthPos = cv::Mat::zeros(3,1,CV_32F);
        cv::Mat pointDepthCov = cv::Mat::zeros(3,3,CV_32F);
    };

    void UncertaintyInTracking::CreateNewMapPoints(ORB_SLAM2::KeyFrame* pKF,cv::Mat &variance, ORB_SLAM2::Tracking &mTracking)
    {
        cv::Mat covPose = mTracking.mCurrentFrame.covpose.clone(); // added pose cov
        cv::Mat trans = mTracking.mCurrentFrame.mTcw.colRange(0,3).rowRange(0,3);
        cv::Mat Twc = mTracking.mCurrentFrame.mTcw.inv();
        cv::Mat AdTwc = cv::Mat::zeros(6,6,CV_32F);
        cv::Mat Rwc = cv::Mat::zeros(3,3,CV_32F);
        cv::Mat twc = cv::Mat::zeros(3,1,CV_32F);
        cv::Mat skewtwc = cv::Mat::zeros(3,3,CV_32F);
        cv::Mat skewtwcRwc = cv::Mat::zeros(3,3,CV_32F);
        Twc.colRange(0,3).rowRange(0,3).copyTo(Rwc);
        Twc.rowRange(0,3).col(3).copyTo(twc);
        skewtwc = skew(twc);
        skewtwcRwc = skewtwc * Rwc;
        Rwc.copyTo(AdTwc.rowRange(0,3).colRange(0,3));
        Rwc.copyTo(AdTwc.rowRange(3,6).colRange(3,6));
        skewtwcRwc.copyTo(AdTwc.rowRange(0,3).colRange(3,6));
        // J_poseinv
        Eigen::Matrix3d Rwc_eigen = ORB_SLAM2::Converter::toMatrix3d(Rwc);    
        Eigen::Vector3d twc_eigen = ORB_SLAM2::Converter::toVector3d(twc);  
        Sophus::SE3 SE3_Rt_inv(Rwc_eigen,twc_eigen);
        Eigen::Matrix<double, 6, 1> se3 = SE3_Rt_inv.log(); 
        Eigen::Vector3d phi = se3.topRows(3);
        Eigen::Vector3d rho = se3.bottomRows(3);
        Eigen::Matrix<double,6,6> J_poseinv;
        J_poseinv.setZero();

        J_poseinv.topLeftCorner(3,3) = hat(phi);
        J_poseinv.topRightCorner(3,1) = hat(rho);
        J_poseinv.bottomRightCorner(3,3) = hat(phi);
        J_poseinv = Eigen::MatrixXd::Identity(6, 6) +  0.5  * J_poseinv;

        cv::Mat J_poseinv_f = Mat6DtoCvMat(J_poseinv);

        const float &fx = mTracking.mCurrentFrame.fx, &fy = mTracking.mCurrentFrame.fy, &cx = mTracking.mCurrentFrame.cx, &cy = mTracking.mCurrentFrame.cy;
        vector<pointTobeFused> pointsTobeFused;
        pointsTobeFused.reserve(mTracking.mCurrentFrame.N);
        vector<pair<float,int> > vDepthIdx;
        vDepthIdx.reserve(mTracking.mCurrentFrame.N);
        for(int i=0; i<mTracking.mCurrentFrame.N; i++)
        {
            float z = mTracking.mCurrentFrame.mvDepth[i];
            if(z>0)
            {
                vDepthIdx.push_back(make_pair(z,i));
            }
        }

        if(!vDepthIdx.empty())
        {
            sort(vDepthIdx.begin(),vDepthIdx.end());

            int nPoints = 0;
            for(size_t j=0; j<vDepthIdx.size();j++)
            {
                int i = vDepthIdx[j].second;

                bool bCreateNew = false;
                bool bFuse = false;

                ORB_SLAM2::MapPoint* pMP = mTracking.mCurrentFrame.mvpMapPoints[i];
                if(!pMP)
                    bCreateNew = true;
                else if(pMP->Observations() < 1)
                {
                    bCreateNew = true;
                    mTracking.mCurrentFrame.mvpMapPoints[i] = static_cast<ORB_SLAM2::MapPoint*>(NULL);
                }
                else if(pMP->iftriangulation && !pMP->ifFused) //added 
                    bFuse = true;
                
                if(bCreateNew || bFuse)
                {
                    
                    cv::Mat x3D = mTracking.mCurrentFrame.UnprojectStereo(i);
                    cv::Mat covXw = cv::Mat::zeros(3,3,CV_32F);

                    const cv::KeyPoint &kp = mTracking.mCurrentFrame.mvKeys[i];
                    const float &uu = kp.pt.x;
                    const float &vv = kp.pt.y; 
                    int u = cvRound(kp.pt.x);
                    int v = cvRound(kp.pt.y);
                    if(u>=0 && u<variance.cols && v>=0 && v<variance.rows) 
                    {
                        float zC   = vDepthIdx[j].first;
                        float varz = variance.at<float>(v, u);

                        cv::Mat covXc = cv::Mat::zeros(3,3,CV_32F);

                        cv::Mat pXc_U = cv::Mat::zeros(3,3,CV_32F);
                        pXc_U.at<float>(0,0) = zC/fx;
                        pXc_U.at<float>(1,1) = zC/fy;
                        pXc_U.at<float>(2,2) = 1.0f;
                        cv::Mat covU = 1.5 * cv::Mat::eye(3,3,CV_32F);
                        cv::Mat pXc_z = cv::Mat::zeros(3,1,CV_32F);
                        pXc_z.at<float>(0,0) = (uu - cx) / fx;
                        pXc_z.at<float>(1,0) = (vv - cy) / fy;
                        pXc_z.at<float>(2,0) = 1.0;

                        covXc = pXc_U*covU*pXc_U.t() + pXc_z * varz * pXc_z.t();
                        cv::Mat covPoseInv = cv::Mat::zeros(6,6,CV_32F);
                        covPoseInv = AdTwc * covPose * AdTwc.t();
                        cv::Mat pXw_poseinv = cv::Mat::zeros(3,6,CV_32F);
                        const float &x = x3D.at<float>(0,0);
                        const float &y = x3D.at<float>(1,0);
                        const float &z = x3D.at<float>(2,0);
                        cv::Mat pXw_poseinv_initial = (cv::Mat_<float>(3,6) <<
                        0,  z, -y, 1, 0, 0,
                        -z,  0,  x, 0, 1, 0,
                        y, -x,  0, 0, 0, 1
                        );
                        pXw_poseinv = pXw_poseinv_initial * J_poseinv_f;
                        
                        covXw = trans*covXc*trans.t() + pXw_poseinv*covPoseInv*pXw_poseinv.t();
                    }

                    if(bCreateNew)
                    {
                        ORB_SLAM2::MapPoint * pNewMP =new MapPoint(x3D,pKF,mTracking.mpMap);

                        pNewMP->SetCovtri3(covXw);
                        pNewMP->iftriangulation = false;

                        pNewMP->AddObservation(pKF,i);
                        pKF->AddMapPoint(pNewMP,i);
                        pNewMP->ComputeDistinctiveDescriptors();
                        pNewMP->UpdateNormalAndDepth();
                        mTracking.mpMap->AddMapPoint(pNewMP);

                        mTracking.mCurrentFrame.mvpMapPoints[i]=pNewMP;
                        nPoints++;
                    }
                    else if(bFuse)
                    {
                        pointTobeFused pointi;
                        pointi.pointIndex = i;
                        pointi.pointDepthPos = x3D.clone();
                        pointi.pointDepthCov = covXw.clone();
                        pointsTobeFused.push_back(pointi);
                    }
                    else
                    {
                        nPoints++;
                    }
                }
                if(vDepthIdx[j].first>mTracking.mThDepth && nPoints>100)
                    break;
            }
        }
        const int NFuse = pointsTobeFused.size();
        PSOParams pso;
        pso.swarm_size =  50;
        pso.max_iters = 100;
        vector<float> best_ws(NFuse);
        if(NFuse > 50)
        {   
            #pragma omp parallel for schedule(dynamic)
            for(unsigned int i = 0; i < NFuse; ++i){
                pointTobeFused thisPoint = pointsTobeFused[i];
                int index = thisPoint.pointIndex;
                ORB_SLAM2::MapPoint *pMP = mTracking.mCurrentFrame.mvpMapPoints[i];
                if(pMP == NULL)
                    continue;
                cv::Mat x1(3,1,CV_32F);
                x1 = pMP->GetWorldPos();
                cv::Mat x2(3,1,CV_32F);
                x2 = thisPoint.pointDepthPos.clone();
                cv::Mat P1(3,3,CV_32F);
                P1 = pMP->GetCovtri3();
                cv::Mat P2(3,3,CV_32F);
                P2 = thisPoint.pointDepthCov.clone();
                best_ws[i] = optimizeWbyPSO(x1,x2,P1,P2,pso);
            }
        }

        else
        {
            for(unsigned int i = 0; i < NFuse; ++i){
                pointTobeFused thisPoint = pointsTobeFused[i];
                int index = thisPoint.pointIndex;
                ORB_SLAM2::MapPoint *pMP = mTracking.mCurrentFrame.mvpMapPoints[i];
                if(pMP == NULL)
                    continue;
                cv::Mat x1(3,1,CV_32F);
                x1 = pMP->GetWorldPos();
                cv::Mat x2(3,1,CV_32F);
                x2 = thisPoint.pointDepthPos.clone();
                cv::Mat P1(3,3,CV_32F);
                P1 = pMP->GetCovtri3();
                cv::Mat P2(3,3,CV_32F);
                P2 = thisPoint.pointDepthCov.clone();
                best_ws[i] = optimizeWbyPSO(x1,x2,P1,P2,pso);
            }
        }
        for(int i = 0; i < NFuse; ++i)
        {
            pointTobeFused pi = pointsTobeFused[i];
            float wi = best_ws[i];
            int index = pi.pointIndex;
            ORB_SLAM2::MapPoint *pMP = mTracking.mCurrentFrame.mvpMapPoints[i];
            if(pMP == NULL)
                continue;
            cv::Mat x1(3,1,CV_32F);
            x1 = pMP->GetWorldPos();
            cv::Mat x2(3,1,CV_32F);
            x2 = pi.pointDepthPos.clone();
            cv::Mat P1(3,3,CV_32F);
            P1 = pMP->GetCovtri3();
            cv::Mat P2(3,3,CV_32F);
            P2 = pi.pointDepthCov.clone();
            cv::Mat xf(3,1,CV_32F);
            cv::Mat Pf(3,3,CV_32F);
            CIFilter(x1,x2,P1,P2,wi,xf,Pf);
            pMP->SetWorldPos(xf);
            pMP->SetCovtri3(Pf);
        }
    }

    struct lineTobeFused
    {
        int lineIndex = -1;
        Eigen::Matrix<double,6,1> lineDepPose = Eigen::Matrix<double,6,1>::Zero();
        cv::Mat lineDepCov = cv::Mat::zeros(6,6,CV_32F);
    };
    void UncertaintyInTracking::CreateNewMapLines(ORB_SLAM2::KeyFrame* pKF,cv::Mat &variance, ORB_SLAM2::Tracking &mTracking)
    {
        cv::Mat covPose = mTracking.mCurrentFrame.covpose.clone();
        cv::Mat trans = mTracking.mCurrentFrame.mTcw.colRange(0,3).rowRange(0,3);
        cv::Mat Twc = mTracking.mCurrentFrame.mTcw.inv();
        cv::Mat AdTwc = cv::Mat::zeros(6,6,CV_32F);
        cv::Mat Rwc = cv::Mat::zeros(3,3,CV_32F);
        cv::Mat twc = cv::Mat::zeros(3,1,CV_32F);
        cv::Mat skewtwc = cv::Mat::zeros(3,3,CV_32F);
        cv::Mat skewtwcRwc = cv::Mat::zeros(3,3,CV_32F);
        Twc.colRange(0,3).rowRange(0,3).copyTo(Rwc);
        Twc.rowRange(0,3).col(3).copyTo(twc);
        skewtwc = skew(twc);
        skewtwcRwc = skewtwc * Rwc;
        Rwc.copyTo(AdTwc.rowRange(0,3).colRange(0,3));
        Rwc.copyTo(AdTwc.rowRange(3,6).colRange(3,6));
        skewtwcRwc.copyTo(AdTwc.rowRange(0,3).colRange(3,6));
        // J_poseinv
        Eigen::Matrix3d Rwc_eigen = ORB_SLAM2::Converter::toMatrix3d(Rwc);    
        Eigen::Vector3d twc_eigen = ORB_SLAM2::Converter::toVector3d(twc);  
        Sophus::SE3 SE3_Rt_inv(Rwc_eigen,twc_eigen);
        Eigen::Matrix<double, 6, 1> se3 = SE3_Rt_inv.log(); 
        Eigen::Vector3d phi = se3.topRows(3);
        Eigen::Vector3d rho = se3.bottomRows(3);
        Eigen::Matrix<double,6,6> J_poseinv;
        J_poseinv.setZero();

        J_poseinv.topLeftCorner(3,3) = hat(phi);
        J_poseinv.topRightCorner(3,1) = hat(rho);
        J_poseinv.bottomRightCorner(3,3) = hat(phi);
        J_poseinv = Eigen::MatrixXd::Identity(6, 6) +  0.5  * J_poseinv;

        cv::Mat J_poseinv_f = Mat6DtoCvMat(J_poseinv);

        const float &fx = mTracking.mCurrentFrame.fx, &fy = mTracking.mCurrentFrame.fy, &cx = mTracking.mCurrentFrame.cx, &cy = mTracking.mCurrentFrame.cy;
        vector<lineTobeFused> linesTobeFused;
        linesTobeFused.reserve(mTracking.mCurrentFrame.NL);
        vector<pair<float,int> > vLineDepthIdx;
        vLineDepthIdx.reserve(mTracking.mCurrentFrame.NL);
        for(int i=0; i<mTracking.mCurrentFrame.NL; i++)
        {
            const float zS = mTracking.mCurrentFrame.mvDepthLineStart[i];
            const float zE = mTracking.mCurrentFrame.mvDepthLineEnd[i];
            if( (zS>0) && (zE>0) )
            {
                const float z  = std::max(zS,zE);
                vLineDepthIdx.push_back(make_pair(z,i));
            }
        }

        if(!vLineDepthIdx.empty())
        {
            sort(vLineDepthIdx.begin(),vLineDepthIdx.end());

            int nLines = 0;
            int newlines = 0;
            for(size_t j=0; j<vLineDepthIdx.size();j++)
            {
                int i = vLineDepthIdx[j].second;
                bool bCreateNew = false;
                bool bFuse = false;

                const float& zSC = mTracking.mCurrentFrame.mvDepthLineStart[i];
                const float& zEC = mTracking.mCurrentFrame.mvDepthLineEnd[i];

                ORB_SLAM2::MapLine* pML = mTracking.mCurrentFrame.mvpMapLines[i];
                if(!pML)
                {
                    bCreateNew = true;
                }
                else if(pML->Observations()<1)
                {
                    bCreateNew = true;
                    mTracking.mCurrentFrame.mvpMapLines[i] = static_cast<ORB_SLAM2::MapLine*>(NULL);
                }
                else if(pML->iftriangulation && !pML->ifFused)
                    bFuse = true;
                
                if(bCreateNew || bFuse)
                {
                    Eigen::Vector3d x3DS, x3DE;
                    bool unproject = false;
                    unproject = mTracking.mCurrentFrame.UnprojectStereoLine(i,x3DS,x3DE);
                    cv::Mat startXw =(cv::Mat_<float>(3,1)<<0,0,0);
                    cv::Mat endXw =(cv::Mat_<float>(3,1)<<0,0,0);
                    cv::Mat covXwS = cv::Mat::zeros(3,3,CV_32F);
                    cv::Mat covXwE = cv::Mat::zeros(3,3,CV_32F);
                    if(unproject)
                    {
                        Eigen::Vector3d v = x3DE - x3DS;
                        v/=v.norm();
                        Eigen::Vector3d n = x3DS.cross(x3DE);
                        n/=n.norm();
                        Vector6d Plucker;
                        Plucker.segment<3>(0) = n;
                        Plucker.segment<3>(3) = v;

                        float u_start = mTracking.mCurrentFrame.mvKeyLinesUn[i].startPointX;
                        float v_start = mTracking.mCurrentFrame.mvKeyLinesUn[i].startPointY;
                        float u_end = mTracking.mCurrentFrame.mvKeyLinesUn[i].endPointX;
                        float v_end = mTracking.mCurrentFrame.mvKeyLinesUn[i].endPointY;

                        float varz_start = variance.at<float>(v_start, u_start);
                        float varz_end = variance.at<float>(v_end, u_end);

                        startXw.at<float>(0,0)=x3DS(0,0);
                        startXw.at<float>(1,0)=x3DS(1,0);
                        startXw.at<float>(2,0)=x3DS(2,0);

                        endXw.at<float>(0,0)=x3DE(0,0);
                        endXw.at<float>(1,0)=x3DE(1,0);
                        endXw.at<float>(2,0)=x3DE(2,0);
                        
                        cv::Mat covstartXc = cv::Mat::zeros(3,3,CV_32F);
                        cv::Mat covendXc = cv::Mat::zeros(3,3,CV_32F);
                        
                        cv::Mat pstartXc_U = cv::Mat::zeros(3,3,CV_32F);
                        pstartXc_U.at<float>(0,0) = zSC/fx;
                        pstartXc_U.at<float>(1,1) = zSC/fy;
                        pstartXc_U.at<float>(2,2) = 1.0f;
                        cv::Mat covU = 1.5 * cv::Mat::eye(3,3,CV_32F); // 100 *
                        cv::Mat pstartXc_z = cv::Mat::zeros(3,1,CV_32F);
                        pstartXc_z.at<float>(0,0) = (u_start - cx) / fx;
                        pstartXc_z.at<float>(1,0) = (v_start - cy) / fy;
                        pstartXc_z.at<float>(2,0) = 1.0;

                        covstartXc = pstartXc_U*covU*pstartXc_U.t() + pstartXc_z * varz_start * pstartXc_z.t();

                        cv::Mat pendXc_U = cv::Mat::zeros(3,3,CV_32F);
                        pendXc_U.at<float>(0,0) = zEC/fx;
                        pendXc_U.at<float>(1,1) = zEC/fy;
                        pendXc_U.at<float>(2,2) = 1.0f;
                        cv::Mat pendXc_z = cv::Mat::zeros(3,1,CV_32F);
                        pendXc_z.at<float>(0,0) = (u_end - cx) / fx;
                        pendXc_z.at<float>(1,0) = (v_end - cy) / fy;
                        pendXc_z.at<float>(2,0) = 1.0;

                        covendXc = pendXc_U*covU*pendXc_U.t() + pendXc_z * varz_end * pendXc_z.t();

                        cv::Mat covPoseInv = cv::Mat::zeros(6,6,CV_32F);
                        covPoseInv = AdTwc * covPose * AdTwc.t();
                        cv::Mat pXwS_poseinv = cv::Mat::zeros(3,6,CV_32F);
                        const float &xS = x3DS(0,0);
                        const float &yS = x3DS(1,0);
                        const float &zS = x3DS(2,0);
                        cv::Mat pXwS_poseinv_initial = (cv::Mat_<float>(3,6) <<
                        0,  zS, -yS, 1, 0, 0,
                        -zS,  0,  xS, 0, 1, 0,
                        yS, -xS,  0, 0, 0, 1
                        );
                        pXwS_poseinv = pXwS_poseinv_initial * J_poseinv_f;

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
                        pXwE_poseinv = pXwE_poseinv_initial * J_poseinv_f;
                        covXwE = trans * covendXc * trans.t() + pXwE_poseinv*covPoseInv*pXwE_poseinv.t();
                        
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

                        if(bCreateNew)
                        {
                            ORB_SLAM2::MapLine* pNewLine = new ORB_SLAM2::MapLine(Plucker,x3DS,x3DE,pKF,mTracking.mpMap);
                            pNewLine->SetcovlinePlk(var_line);

                            Vector4d orth = plk_to_orth(Plucker);
                            pNewLine->SetWorldOR(orth);
                            cv::Mat plk(6,1,CV_32F);
                            plk = VD6toCvMat(Plucker);
                            cv::Mat pOR_Plk(4,6,CV_32F);
                            pOR_Plk = jacobianFromPlktoOrth(plk);

                            cv::Mat var_line_or(4,4,CV_32F);
                            var_line_or = pOR_Plk * var_line * pOR_Plk.t();

                            pNewLine->ifFused=true;
                            pNewLine->SetcovlineOR(var_line_or);

                            pNewLine->AddObservation(pKF,i);
                            pKF->AddMapLine(pNewLine,i);
                            pNewLine->ComputeDistinctiveDescriptors();
                            pNewLine->UpdateNormalAndDepth();
                            mTracking.mpMap->AddMapLine(pNewLine);

                            mTracking.mCurrentFrame.mvpMapLines[i]=pNewLine;
                            nLines++;
                            newlines++;
                            
                        }

                        else if(bFuse)
                        {
                            lineTobeFused linei;
                            linei.lineIndex = i;
                            linei.lineDepPose = Plucker;
                            linei.lineDepCov = var_line.clone();
                            linesTobeFused.push_back(linei);
                        }   
                        else
                        {
                            nLines++;
                        }
                    }
                }
                if(vLineDepthIdx[j].first>mTracking.mThDepth && nLines>100)
                        break;
            }
        }

        const int NFuse = linesTobeFused.size();
        PSOParams pso;
        pso.swarm_size = 50;
        pso.max_iters = 100;
        vector<float> best_ws(NFuse);
        if(NFuse > 50)
        {
            #pragma omp parallel for schedule(dynamic)
            for(unsigned int i = 0; i < NFuse; ++i){
                lineTobeFused thisLine = linesTobeFused[i];
                int index = thisLine.lineIndex;
                ORB_SLAM2::MapLine *pML = mTracking.mCurrentFrame.mvpMapLines[i];
                if(pML == NULL)
                    continue;
                cv::Mat x1(6,1,CV_32F);
                x1 = VD6toCvMat(pML->GetWorldPlk());
                cv::Mat x2(6,1,CV_32F);
                x2 = VD6toCvMat(thisLine.lineDepPose);
                cv::Mat P1(6,6,CV_32F);
                P1 = pML->GetcovlinePlk();
                cv::Mat P2(6,6,CV_32F);
                P2 = thisLine.lineDepCov.clone();
                best_ws[i] = optimizeWbyPSO(x1,x2,P1,P2,pso);
            }
        }
        else
        {
            for(unsigned int i = 0; i < NFuse; ++i){
                lineTobeFused thisLine = linesTobeFused[i];
                int index = thisLine.lineIndex;
                ORB_SLAM2::MapLine *pML = mTracking.mCurrentFrame.mvpMapLines[i];
                if(pML == NULL)
                    continue;
                cv::Mat x1(6,1,CV_32F);
                x1 = VD6toCvMat(pML->GetWorldPlk());
                cv::Mat x2(6,1,CV_32F);
                x2 = VD6toCvMat(thisLine.lineDepPose);
                cv::Mat P1(6,6,CV_32F);
                P1 = pML->GetcovlinePlk();
                cv::Mat P2(6,6,CV_32F);
                P2 = thisLine.lineDepCov.clone();
                best_ws[i] = optimizeWbyPSO(x1,x2,P1,P2,pso);
            }
        }
        for(int i = 0; i < NFuse; ++i)
        {
            lineTobeFused li = linesTobeFused[i];
            float wi = best_ws[i];
            int index = li.lineIndex;
            ORB_SLAM2::MapLine *pML = mTracking.mCurrentFrame.mvpMapLines[i];
            if(pML == NULL)
                continue;
            cv::Mat x1(6,1,CV_32F);
            x1 = VD6toCvMat(pML->GetWorldPlk());
            cv::Mat x2(6,1,CV_32F);
            x2 = VD6toCvMat(li.lineDepPose);
            cv::Mat P1(6,6,CV_32F);
            P1 = pML->GetcovlinePlk();
            cv::Mat P2(6,6,CV_32F);
            P2 = li.lineDepCov.clone();
            cv::Mat xf(6,1,CV_32F);
            cv::Mat Pf(6,6,CV_32F);
            CIFilter(x1,x2,P1,P2,wi,xf,Pf);
            Eigen::Matrix<double,6,1> xf_eigen;
            xf_eigen<<xf.at<float>(0,0), xf.at<float>(1,0), xf.at<float>(2,0), xf.at<float>(3,0), xf.at<float>(4,0), xf.at<float>(5,0);
            pML->SetWorldPlk(xf_eigen);
            pML->SetcovlinePlk(Pf);
        }
    }
}