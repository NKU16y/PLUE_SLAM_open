## PLUE-SLAM: An RGB-D SLAM System Integrating Uncertainty Estimation of Point and Line Features
## 1. Introduction
<div style="text-align: justify;">
The code for the uncertainty estimation module in PLUE-SLAM, a point–line RGB-D SLAM method built on the ORB-SLAM2 framework.

Currently, the uncertainty-related code in the front-end tracking is already available, including the initialization, feature and pose uncertainty estimation during tracking, and the uncertainty initialization for newly created features with CI-filter fusion in the tracking phase. We will open-source the remaining code after the paper is accepted.

This code must be integrated into [ORB-SLAM2](https://github.com/raulmur/ORB_SLAM2) and also needs to implement the line feature association and optimization. You can refer to [PL-VINS](https://github.com/cnqiangfu/PL-VINS) implementation and the [G2O documentation](https://github.com/RainerKuemmerle/g2o).



## 2 License
PLUE-SLAM was developed at the Nankai University of Tianjin, China.
The open-source version is licensed under the GNU General Public License
Version 3 (GPLv3).
