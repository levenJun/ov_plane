/*
 * ov_plane: Monocular Visual-Inertial Odometry with Planar Regularities
 * Copyright (C) 2022-2023 Chuchu Chen
 * Copyright (C) 2022-2023 Patrick Geneva
 * Copyright (C) 2022-2023 Guoquan Huang
 *
 * OpenVINS: An Open Platform for Visual-Inertial Research
 * Copyright (C) 2018-2023 Patrick Geneva
 * Copyright (C) 2018-2023 Guoquan Huang
 * Copyright (C) 2018-2023 OpenVINS Contributors
 * Copyright (C) 2018-2019 Kevin Eckenhoff
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#include "PlaneFitting.h"

#include <boost/date_time/posix_time/posix_time.hpp>
#include <ceres/ceres.h>

#include "ceres/Factor_ImageReprojCalib.h"
#include "ceres/Factor_PointOnPlane.h"
#include "ceres/State_JPLQuatLocal.h"
#include "feat/Feature.h"
#include "utils/colors.h"
#include "utils/print.h"
#include "utils/quat_ops.h"

using namespace ov_core;
using namespace ov_plane;


//拟合平面:求解Ax=b的线性方程
  //线性方程有效性check:要求A的svd分解奇异值,s(0)/s(2) < 50.  就是保证A矩阵的满秩?
  //平面方程有效性check:原点到平面的距离要求至少大于5厘米
//feats:输入用于拟合平面的点集
//abcd:平面方程参数
//cond_thresh:线性方程chec阈值
//cond_check:平面方程距离check阈值
bool PlaneFitting::fit_plane(const std::vector<std::shared_ptr<ov_core::Feature>> &feats, Eigen::Vector4d &abcd, double cond_thresh,
                             bool cond_check) {

  // Check whether we have enough constraints
  if (feats.size() < 3) {
    // PRINT_DEBUG("[PLANE-FIT]: Not having enough constraint for plane fitting! (%d )\n", feats.size());
    return false;
  }

  // Linear system
  // 平面方程:P^T * n + 1 = 0
  // 可以求解线性方程An=b得到齐次法向n
  // 线性方程有效性check:A的svd分解奇异值,s(0)/s(2) < 50.  就是保证A矩阵的满秩?
  // 求解的平面方程有效性check:原点到平面的距离要求至少大于5厘米
  Eigen::MatrixXd A = Eigen::MatrixXd::Zero((int)feats.size(), 3);
  Eigen::VectorXd b = -Eigen::VectorXd::Ones((int)feats.size(), 1);
  for (size_t i = 0; i < feats.size(); i++) {
    A.row(i) = feats.at(i)->p_FinG.transpose();
  }

  // Check condition number to avoid singularity
  if (cond_check) {
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A);
    double cond = svd.singularValues()(0) / svd.singularValues()(svd.singularValues().size() - 1);
    if (cond > cond_thresh) {
      // PRINT_DEBUG("[PLANE-FIT]: The condition number is too big!! (%.2f > %.2f)\n", cond, cond_thresh);
      return false;
    }
  }

  // Solve plane a b c d by QR decomposition
  // ax + by + cz + d = 0
  abcd.head(3) = A.colPivHouseholderQr().solve(b);
  // d = 1.0
  abcd(3) = 1.0;
  // Divide the whole vector by the norm of the normal direction of the plane
  abcd /= abcd.head(3).norm();//将法向还原为单位法向

  // Check if the plane is invalid (depth of plane near zero)
  double dist_thresh = 0.02;
  Eigen::Vector3d cp = -abcd.head(3) * abcd(3); //abcd.head(3)模长本来就是1,所以cp模长结果应该就是截距d,即原点到面的距离.
  return (cp.norm() > dist_thresh);
}

//随机采样5点拟合平面:内点距离阈值是5厘米
  //有效平面要求内点数至少10个内点 && 内点数占比至少0.8
//记录最佳内点集合为best_inliers,最小平均err为best_error,并用内点集重新拟合平面
//params:
  //feats:输入待拟合点集
  //plane_abcd:拟合的平面结果
  //min_inlier_num:内点数最小阈值
  //max_plane_solver_condition_number:An=b,线性方程解平面时,A矩阵的check阈值
bool PlaneFitting::plane_fitting(std::vector<std::shared_ptr<ov_core::Feature>> &feats, Eigen::Vector4d &plane_abcd, int min_inlier_num,
                                 double max_plane_solver_condition_number) {
  // RANSAC params for plane fitting
  // TODO: read these from parameter file....
  const int ransac_solver_feat_num = 5;
  const int max_iter_num = 200;
  const double min_inlier_ratio = 0.80;
  const double max_error_threshold = 0.05;
  const double min_distance_between_points = 0.05;
  const size_t min_feat_on_plane_num_threshold = std::max(min_inlier_num, (int)((double)feats.size() * min_inlier_ratio));
  std::mt19937 rand_gen(8888);

  // If the features are too few, we skip them
  if ((int)feats.size() < min_inlier_num) {
    PRINT_DEBUG(BOLDRED "[PLANE-FIT]: failed not enough feature (%zu feats)\n" RESET, feats.size());
    return false;
  }

  // Solve by ransace if we have enough features
  //随机采样5点拟合平面:内点距离阈值是5厘米
    //有效平面要求内点数至少10个内点 && 内点数占比至少0.8
  //记录最佳内点集合为best_inliers,最小平均err为best_error
  double best_error = -1;
  std::vector<std::shared_ptr<ov_core::Feature>> best_inliers;
  for (size_t n = 0; n < max_iter_num; n++) {

    // Create a random set of features
    // 输入点集,随机采样5个点:5个点两两间距离都必须大于5厘米
    std::vector<std::shared_ptr<ov_core::Feature>> feat_vec_copy = feats;
    std::vector<std::shared_ptr<ov_core::Feature>> ransac_set;            //随机采样的5点结果
    std::shuffle(feat_vec_copy.begin(), feat_vec_copy.end(), rand_gen);

    // Loop until we have enough points or when we run out of option
    auto it = feat_vec_copy.begin();
    while (ransac_set.size() < ransac_solver_feat_num && it != feat_vec_copy.end()) {

      // Push back directly for the first one
      if (ransac_set.empty()) {
        ransac_set.push_back(*it);
      } else {
        // Check distance between this point and the current set
        bool good_feat = true;
        Eigen::Vector3d p_FinG = (*it)->p_FinG;
        for (const auto &pf : ransac_set) {
          if ((pf->p_FinG - p_FinG).norm() < min_distance_between_points) {
            good_feat = false;
            break;
          }
        }
        // If pass distance check, add it to the ransac set
        if (good_feat) {
          ransac_set.push_back(*it);
        }
      }
      it++;
    }

    // If not enough features the just return
    if (ransac_set.size() != ransac_solver_feat_num) {
      PRINT_INFO(BOLDRED "[PLANE-FIT]: failed not enough inliers (%zu feats)\n" RESET, ransac_set.size());
      return false;
    }

    // Try to solve the plane when we have enough features
    //5点拟合平面:求解Ax=b的线性方程
      //线性方程有效性check:要求A的svd分解奇异值,s(0)/s(2) < 50.  就是保证A矩阵的满秩?
      //平面方程有效性check:原点到平面的距离要求至少大于5厘米    
    
    //内点统计:
      //内点阈值-点面距离5厘米;
      //内点数check:至少10个内点 && 内点数占比至少0.8
    if (fit_plane(ransac_set, plane_abcd, max_plane_solver_condition_number)) {

      // Check the other p_FinG and check the number of iniliers
      double inlier_avg_error = 0.0;
      std::vector<std::shared_ptr<ov_core::Feature>> inliers;
      for (auto &feat : feats) {
        double error = point_to_plane_distance(feat->p_FinG, plane_abcd);
        if (std::abs(error) < max_error_threshold) {
          inliers.push_back(feat);
          inlier_avg_error += std::abs(error);
        }
      }
      inlier_avg_error /= (double)inliers.size();

      // If pass inlier number threshold and error threshold then the plane is good to be accepted
      // The set is better if it has more inliers or if it has the same number and smaller error!
      bool valid_set = (inliers.size() > min_feat_on_plane_num_threshold && inlier_avg_error < max_error_threshold);
      bool better_set =
          ((best_inliers.size() < inliers.size()) || (best_inliers.size() == inliers.size() && inlier_avg_error < best_error));
      if (valid_set && better_set) {
        best_inliers = inliers;
        best_error = inlier_avg_error;
      }
    }
  }

  // Check that we have a good set of inliers
  if (!best_inliers.empty()) {

    // Further optimize the initial value using the inlier set
    if (fit_plane(best_inliers, plane_abcd, max_plane_solver_condition_number, false)) {

      // Calculate measurement sigma
      Eigen::ArrayXd errors = Eigen::ArrayXd::Zero((int)best_inliers.size(), 1);
      for (size_t i = 0; i < best_inliers.size(); i++) {
        errors(i) = PlaneFitting::point_to_plane_distance(best_inliers.at(i)->p_FinG, plane_abcd);
      }
      double inlier_std = std::sqrt((errors - errors.mean()).square().sum() / (double)(errors.size() - 1));
      double inlier_err = errors.abs().mean();

      // Debug print
      PRINT_INFO(BOLDCYAN "[PLANE-FIT]: success tri %.3f +- %.3f avg dist (%zu of %zu)\n" RESET, inlier_err, inlier_std,
                 best_inliers.size(), feats.size());
      feats = best_inliers;
      return true;
    }
  }

  // All bad :(
  PRINT_INFO(BOLDRED "[PLANE-FIT]: ransac failed | %.3f cost | %zu inliers | %zu feats |\n" RESET, best_error, best_inliers.size(),
             feats.size());
  return false;
}

// 初始化单个大平面参数cp后,利用单个大平面的观测小点,构建点的BA残差和点面距离残差,对相关的面特征参数cp,点特征坐标,相机pose(fixed),相机内参和外参(fixed)作联合优化
// 观测:
    // (一)点BA观测:
    // 平面观测:
// 涉及状态:
    // (Π1)单个大平面参数<ceres_cp>: 正常可优化状态,(但是fix_plane=true开关可以强制设为fixed)
    // (x1)追踪到单个大平面的点特征<ceres_vars_feat>: 正常可优化状态.(但是BA观测数为0的被强制设置为fixed)
    // (x2)追踪到大平面的相机帧pose<ceres_vars_ori/pos>: fixed
    // (x3-1)相机外参<ceres_vars_calib_cam2imu_ori/pos>: fixed
    // (x3-2)相机内参<ceres_vars_calib_cam_intrinsics>: fixed
// 优化问题构建:
// 执行完最小二乘优化,得到优化后的:1)大平面参数--cp_inG.2)小点修正坐标--feats[i]->p_FinG,3)满足距离要求的内点集合替换原输入点集--feats

//param:
  //feats:单个大平面的观测小点集合作为输入. !!优化完成后,无效的外点会被剔除!!
  //cp_inG:单个平面初始化参数:-法向n*截距d !!优化完成后,被替换为优化后平面参数!!
  //clonesCAM:
  // 
bool PlaneFitting::optimize_plane(std::vector<std::shared_ptr<ov_core::Feature>> &feats, Eigen::Vector3d &cp_inG,
                                  std::unordered_map<size_t, std::unordered_map<double, ov_core::FeatureInitializer::ClonePose>> &clonesCAM,
                                  double sigma_px_norm, double sigma_c, bool fix_plane, const Eigen::VectorXd &stateI,
                                  const Eigen::VectorXd &calib0) {

  // Params for plane fitting
  // TODO: read these from parameter file....
  const double min_inlier_ratio = 0.80;
  const double max_error_threshold = 0.03;
  const double slam_inflation = 2.00;
  const size_t min_feat_on_plane_num_threshold = std::max(4, (int)((double)feats.size() * min_inlier_ratio));

  // Ensure we have enough features
  if ((!fix_plane && feats.size() < 4) || (fix_plane && feats.empty())) {
    PRINT_DEBUG(RED "[PLANE-OPT]: failed not enough feature (%zu feats)\n" RESET, feats.size());
    return false;
  }
  assert(stateI.rows() == 7);
  assert(calib0.rows() == 7);
  auto rT0 = boost::posix_time::microsec_clock::local_time();

  // Now lets create the ceres problem to optimize!
  ceres::Problem problem;

  // 3d features in global
  std::map<size_t, int> map_features;   //(x1)存储的是点特征索引idx. {点特征id, 点特征状态在下面缓存列表的指针序号}
  std::vector<double *> ceres_vars_feat;//点特征状态作为ceres优化参数指针,专门缓存.
                                          //具体索引序号idx存储在 map_features 

  // cp plane
  auto *ceres_cp = new double[3];
  ceres_cp[0] = cp_inG(0);
  ceres_cp[1] = cp_inG(1);
  ceres_cp[2] = cp_inG(2);
  problem.AddParameterBlock(ceres_cp, 3);             //(Π1)大平面特征作为估计参数
  if (fix_plane) {
    problem.SetParameterBlockConstant(ceres_cp);      //(Π1)依开关置为fixed
  }

  // FIXED: camera poses for each cam_id, indexed by time
  // 不同帧的相机pose,作为平面观测的状态输入,也作为状态
      //记录相机pose状态在ceres优化参数中的位序
  std::map<size_t, std::map<double, int>> map_states;   //(x2)存储的是相机帧pose的索引idx. 二级key记录--{相机id : {帧stamp : 相机状态在缓存列表的索引idx}}
                                                          //大平面特征可能被不同相机观测到,相机id作为一级key
                                                          //大平面特征可能被同一个相机有多帧观测到,不同帧时间戳作为二级key
                                                          //具体值是,具体相机帧pose状态,在(同一个相机下)ceres优化参数中的位序.用来作为下面指针列表的索引idx
  
  std::vector<double *> ceres_vars_ori;                   //相机帧pose状态作为ceres优化参数指针,专门缓存.
                                                          //具体索引序号idx存储在 map_states 
  std::vector<double *> ceres_vars_pos;

  // FIXED: extrinsic calibration q_ItoC, p_IinC (map from camera id to index)
  std::map<size_t, int> map_calib_cam2imu;              //(x3-1)存储的是相机外参的索引idx. {相机id : 相机外参在缓存列表的索引idx}
  std::vector<double *> ceres_vars_calib_cam2imu_ori;
  std::vector<double *> ceres_vars_calib_cam2imu_pos;

  // FIXED: intrinsic calibration focal, center, distortion (map from camera id to index)
  std::map<size_t, int> map_calib_cam;                  //(x3-2)存储的是相机内参的索引idx. {相机id : 相机内参在缓存列表的索引idx}
  std::vector<double *> ceres_vars_calib_cam_intrinsics;

  // Helper function that will append a new constraint factor to ceres problem
  //总的残差约束有点特征Pw重投影残差,点Pw到面cp距离残差
  //(二)点面残差:下面流程主要在构建点的BA残差,每个点BA残差构建结束时,顺便构建点面距离残差: 
          // 注意!!!!!!!!!点特征有一个视觉BA观测,就添加一个点面残差??
  auto add_constraint = [&](const std::shared_ptr<ov_core::Feature> &feat, double inflation = 1.0) {
    std::vector<double *> factor_params_const;
    factor_params_const.push_back(ceres_vars_feat.at(map_features.at(feat->featid))); //(x1)点特征作为估计状态添加进ceres
    factor_params_const.push_back(ceres_cp);                                          //(Π1)面特征作为估计状态添加进ceres
    auto *factor_const = new Factor_PointOnPlane(inflation * sigma_c);          //(二)点面距离残差
    // ceres::LossFunction *loss_function_const = nullptr;
    ceres::LossFunction *loss_function_const = new ceres::CauchyLoss(1.0);
    problem.AddResidualBlock(factor_const, loss_function_const, factor_params_const);
  };

  // Loop through each feature
  //(一)构建纯点特征的观测方程(和平面特征无关)
  for (auto const &feat : feats) {    //feat:匹配到大平面的单个小特征点

    // Append this feature estimate
    assert(map_features.find(feat->featid) == map_features.end());
    auto *var_feat = new double[3];
    for (int i = 0; i < 3; i++) {
      var_feat[i] = feat->p_FinG(i);
    }
    problem.AddParameterBlock(var_feat, 3);                           //(x1)点特征作为估计状态添加进ceres
    map_features.insert({feat->featid, (int)ceres_vars_feat.size()}); //记录点特征在ceres指针缓存中的位序
    ceres_vars_feat.push_back(var_feat);                              //点特征在ceres指针缓存

    // Count number of measurements
    int ct_meas = 0;
    for (const auto &pair : feat->timestamps)
      ct_meas += (int)feat->timestamps[pair.first].size();

    // For slam feature we don't change their estimate
    if (ct_meas == 0) {
      problem.SetParameterBlockConstant(var_feat);
      add_constraint(feat, slam_inflation);                           //slam特征:fixed
    }

    // Loop through each camera for this feature
    for (auto const &pair : feat->timestamps) {//feat:匹配到大平面的单个小特征点
                                                //pair:单个小特征点对应单个相机所有观测列表

      // State poses
      size_t cam_id = pair.first;
      if (map_states.find(cam_id) == map_states.end()) {
        map_states.insert({cam_id, std::map<double, int>()});
      }

      // Camera extrinsics (identity since poses are camera frame)
      // 相机外参:fixed
      if (map_calib_cam2imu.find(cam_id) == map_calib_cam2imu.end()) {
        auto *var_calib_ori = new double[4];
        auto *var_calib_pos = new double[3];
        for (int i = 0; i < 3; i++) {
          var_calib_ori[i] = 0.0;
          var_calib_pos[i] = 0.0;
        }
        var_calib_ori[3] = 1.0;
        auto ceres_calib_jplquat = new ov_init::State_JPLQuatLocal();
        problem.AddParameterBlock(var_calib_ori, 4, ceres_calib_jplquat);               //(x3-1)相机外参作为估计状态
        problem.AddParameterBlock(var_calib_pos, 3);
        map_calib_cam2imu.insert({cam_id, (int)ceres_vars_calib_cam2imu_ori.size()});   //记录相机外参在ceres指针缓存中的位序
        ceres_vars_calib_cam2imu_ori.push_back(var_calib_ori);                          //相机外参在ceres指针缓存
        ceres_vars_calib_cam2imu_pos.push_back(var_calib_pos);
        problem.SetParameterBlockConstant(var_calib_ori);                               //相机外参置为fixed
        problem.SetParameterBlockConstant(var_calib_pos);
      }

      // Camera intrinsics
      //相机内参:fixed
      bool is_fisheye = false;
      if (map_calib_cam.find(cam_id) == map_calib_cam.end()) {
        auto *var_calib_cam = new double[8];
        var_calib_cam[0] = 1.0;
        var_calib_cam[1] = 1.0;
        for (int i = 2; i < 8; i++)
          var_calib_cam[i] = 0.0;
        problem.AddParameterBlock(var_calib_cam, 8);                                    //(x3-2)相机内参作为估计状态
        map_calib_cam.insert({cam_id, (int)ceres_vars_calib_cam_intrinsics.size()});    //记录相机内参在ceres指针缓存中的位序
        ceres_vars_calib_cam_intrinsics.push_back(var_calib_cam);                       //相机内参在ceres指针缓存
        problem.SetParameterBlockConstant(var_calib_cam);                               //相机内参置为fixed
      }

      // Measurements
      for (size_t m = 0; m < feat->timestamps.at(pair.first).size(); m++) {//feat:匹配到大平面的单个小特征点
                                                                            //pair:单个小特征点对应单个相机所有观测列表
                                                                            //m:  单个小特征单个相机观测列表的第m个具体观测帧

        // Get the position of this clone in the global
        double timestamp = feat->timestamps[pair.first].at(m);
        Eigen::Vector4d q_GtoCi = rot_2_quat(clonesCAM.at(pair.first).at(timestamp).Rot());
        Eigen::Vector3d p_CiinG = clonesCAM.at(pair.first).at(timestamp).pos();

        // Append to ceres problem if we don't have this state
        //相机帧pose作为估计状态
        //fixed
        if (map_states.at(cam_id).find(timestamp) == map_states.at(cam_id).end()) {
          auto *var_ori = new double[4];
          for (int i = 0; i < 4; i++) {
            var_ori[i] = q_GtoCi(i);
          }
          auto *var_pos = new double[3];
          for (int i = 0; i < 3; i++) {
            var_pos[i] = p_CiinG(i);
          }
          auto ceres_jplquat = new ov_init::State_JPLQuatLocal();
          problem.AddParameterBlock(var_ori, 4, ceres_jplquat);                         //(x2)相机帧pose作为估计状态
          problem.AddParameterBlock(var_pos, 3);
          map_states.at(cam_id).insert({timestamp, (int)map_states.at(cam_id).size()}); //记录相机帧pose在ceres中的位序
          ceres_vars_ori.push_back(var_ori);                                            //相机帧pose状态作为ceres优化参数指针,专门记录下
          ceres_vars_pos.push_back(var_pos);
          problem.SetParameterBlockConstant(var_ori);                                   //设置fixed不作优化
          problem.SetParameterBlockConstant(var_pos);
        }

        // Get measurement
        Eigen::Vector2d uv_norm = feat->uvs_norm.at(pair.first).at(m).block(0, 0, 2, 1).cast<double>();//单点特征在相机i,帧m下的像素观测值

        // REPROJECTION: Factor parameters it is a function of
        std::vector<double *> factor_params;
        factor_params.push_back(ceres_vars_ori.at(map_states.at(cam_id).at(timestamp)));          //(x2)相机帧pose作为估计状态
        factor_params.push_back(ceres_vars_pos.at(map_states.at(cam_id).at(timestamp)));
        factor_params.push_back(ceres_vars_feat.at(map_features.at(feat->featid)));               //(x1)点特征作为估计状态添
        factor_params.push_back(ceres_vars_calib_cam2imu_ori.at(map_calib_cam2imu.at(cam_id)));   //(x3-1)相机外参作为估计状态
        factor_params.push_back(ceres_vars_calib_cam2imu_pos.at(map_calib_cam2imu.at(cam_id)));
        factor_params.push_back(ceres_vars_calib_cam_intrinsics.at(map_calib_cam.at(cam_id)));    //(x3-2)相机内参作为估计状态
        auto *factor_pinhole = new ov_init::Factor_ImageReprojCalib(uv_norm, sigma_px_norm, is_fisheye);
        // ceres::LossFunction *loss_function = nullptr;
        ceres::LossFunction *loss_function = new ceres::CauchyLoss(1.0);
        problem.AddResidualBlock(factor_pinhole, loss_function, factor_params);                   //(一)点特征Pw投影到带畸变的像素uv,与观测量构成的观测残差!
                                                                                                  //(一)相关状态有点特征Pw,相机pose,外参Poes,内参

        // CONSTRAINT: Factor parameters it is a function of
        // TODO: is this better or good to do single or multiple of these?
        add_constraint(feat);   //(二)点面残差
      }
    }
  }

  // Set the optimization settings
  // NOTE: We use dense schur since after eliminating features we have a dense problem
  // NOTE: http://ceres-solver.org/solving_faqs.html#solving
  ceres::Solver::Options options;
  options.linear_solver_type = ceres::DENSE_SCHUR;
  options.trust_region_strategy_type = ceres::DOGLEG;
  options.num_threads = 1; // should be 1 for SIM repeatability
  options.max_num_iterations = 12;
  // options.minimizer_progress_to_stdout = true;
  // options.function_tolerance = 1e-5;
  // options.gradient_tolerance = 1e-4 * options.function_tolerance;

  // Optimize the ceres graph and return if converged...
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  auto rT1 = boost::posix_time::microsec_clock::local_time();
  double time_ms = (double)(rT1 - rT0).total_microseconds() * 1e-3;

  // Free all data structures
  auto cleanup_ceres_memory = [&]() {
    for (auto &ptr : ceres_vars_feat)
      delete[] ptr;
    delete[] ceres_cp;
    for (auto &ptr : ceres_vars_ori)
      delete[] ptr;
    for (auto &ptr : ceres_vars_pos)
      delete[] ptr;
    for (auto &ptr : ceres_vars_calib_cam2imu_ori)
      delete[] ptr;
    for (auto &ptr : ceres_vars_calib_cam2imu_pos)
      delete[] ptr;
    for (auto &ptr : ceres_vars_calib_cam_intrinsics)
      delete[] ptr;
  };

  // For debug
  size_t num_states = 0;
  for (auto const &states : map_states)
    num_states += states.second.size();

  // Finally return failure if we have not converged
  // Else we can recover the optimized plane and features!
  if (summary.termination_type != ceres::CONVERGENCE) {
    PRINT_INFO(BOLDRED "[PLANE-OPT]: failed %d iter %.1f ms | %zu states, %zu feats | %d param and %d res | cost %.4e => %.4e |\n" RESET,
               (int)summary.iterations.size(), time_ms, num_states, map_features.size(), summary.num_parameters, summary.num_residuals,
               summary.initial_cost, summary.final_cost);
    cleanup_ceres_memory();
    return false;
  }

  // Update estimate of cp plane  
  // 执行完最小二乘优化,得到并返回优化后的:1)单个大平面参数--cp_inG.2)小点修正坐标--feats[i]->p_FinG,3)满足距离要求的内点集合替换原输入点集--feats
  Eigen::Vector3d cp_inG_before = cp_inG;
  cp_inG(0) = ceres_cp[0];
  cp_inG(1) = ceres_cp[1];
  cp_inG(2) = ceres_cp[2];
  Eigen::Vector3d cp_inG_diff = (cp_inG - cp_inG_before);
  // std::cout << "cp diff -> " << cp_inG_diff.transpose() << std::endl;
  Eigen::Vector4d plane_abcd;
  plane_abcd.head(3) = cp_inG / cp_inG.norm();
  plane_abcd(3) = -cp_inG.norm();

  // IMU historical clone
  Eigen::Matrix3d R_GtoI = quat_2_Rot(stateI.block(0, 0, 4, 1));
  Eigen::Vector3d p_IinG = stateI.block(4, 0, 3, 1);

  // Calibration
  Eigen::Matrix3d R_ItoC = quat_2_Rot(calib0.block(0, 0, 4, 1));
  Eigen::Vector3d p_IinC = calib0.block(4, 0, 3, 1);

  // Convert current CAMERA position relative to global
  Eigen::Matrix3d R_GtoCi = R_ItoC * R_GtoI;
  Eigen::Vector3d p_CiinG = p_IinG - R_GtoCi.transpose() * p_IinC;

  // Update the estimate of p_FinG for all features
  std::vector<std::shared_ptr<ov_core::Feature>> inliers;
  double diff_avg = 0.0;
  for (auto const &feat : feats) {

    // Recover the pose estimate
    Eigen::Vector3d p_FinG_before = feat->p_FinG;
    Eigen::Vector3d p_FinG_after;
    p_FinG_after << ceres_vars_feat.at(map_features.at(feat->featid))[0], ceres_vars_feat.at(map_features.at(feat->featid))[1],
        ceres_vars_feat.at(map_features.at(feat->featid))[2];

    // Check that point to plane distance is good to go
    double error = point_to_plane_distance(feat->p_FinG, plane_abcd);
    if (std::abs(error) >= max_error_threshold)
      continue;

    // Skip if invalid feature
    if (std::isnan(p_FinG_after.norm()))
      continue;

    // Check if the feature is in front of the camera (non-SLAM feats)
    Eigen::MatrixXd p_FinCi = R_GtoCi * (p_FinG_after - p_CiinG);
    if (p_FinCi(2, 0) < 0.1)
      continue;

    // Finally record the updates position
    feat->p_FinG = p_FinG_after;
    inliers.push_back(feat);
    diff_avg += (feat->p_FinG - p_FinG_before).norm();
    // std::cout << feat->featid << " diff -> " << (feat->p_FinG - p_FinG_before).transpose() << std::endl;
  }
  if (!inliers.empty()) {
    diff_avg /= (double)inliers.size();
  }

  // Return if not enough inliers
  if ((!fix_plane && feats.size() != 1 && inliers.size() < min_feat_on_plane_num_threshold) ||
      (fix_plane && feats.size() != 1 && inliers.size() < min_feat_on_plane_num_threshold) ||
      (fix_plane && feats.size() == 1 && inliers.empty())) {
    PRINT_INFO(BOLDRED "[PLANE-OPT]: failed %d iter %.1f ms | only %zu inliers found |\n" RESET, (int)summary.iterations.size(), time_ms,
               inliers.size());
    cleanup_ceres_memory();
    return false;
  }

  // Calculate measurement sigma
  Eigen::ArrayXd errors = Eigen::ArrayXd::Zero((int)inliers.size(), 1);
  for (size_t i = 0; i < inliers.size(); i++) {
    errors(i) = PlaneFitting::point_to_plane_distance(inliers.at(i)->p_FinG, plane_abcd);
  }
  double inlier_std = std::sqrt((errors - errors.mean()).square().sum() / (double)(errors.size() - 1));
  double inlier_err = errors.abs().mean();

  // Debug print
  PRINT_DEBUG(BOLDCYAN "[PLANE-OPT]: success opt %.3f +- %.3f avg dist (%zu of %zu)\n" RESET, inlier_err, inlier_std, inliers.size(),
              feats.size());
  PRINT_DEBUG(BOLDCYAN "[PLANE-OPT]: cp_diff = %.3f, %.3f, %.3f | feat diff = %.3f (m) |\n" RESET, cp_inG_diff(0), cp_inG_diff(1),
              cp_inG_diff(2), diff_avg);
  PRINT_INFO(BOLDCYAN "[PLANE-OPT]: success %d iter %.1f ms | %zu states, %zu feats | %d param and %d res | cost %.4e => %.4e |\n" RESET,
             (int)summary.iterations.size(), time_ms, num_states, map_features.size(), summary.num_parameters, summary.num_residuals,
             summary.initial_cost, summary.final_cost);
  feats = inliers;
  cleanup_ceres_memory();
  return true;
}
