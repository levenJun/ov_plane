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

#include "UpdaterPlane.h"
#include "UpdaterHelper.h"

#include "feat/Feature.h"
#include "feat/FeatureInitializer.h"
#include "state/State.h"
#include "state/StateHelper.h"
#include "track_plane/PlaneFitting.h"
#include "types/Landmark.h"
#include "types/LandmarkRepresentation.h"
#include "utils/colors.h"
#include "utils/print.h"
#include "utils/quat_ops.h"

#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/math/distributions/chi_squared.hpp>

using namespace ov_core;
using namespace ov_type;
using namespace ov_plane;

UpdaterPlane::UpdaterPlane(UpdaterOptions &options, ov_core::FeatureInitializerOptions &feat_init_options) : _options(options) {

  // Save our feature initializer
  initializer_feat = std::shared_ptr<ov_core::FeatureInitializer>(new ov_core::FeatureInitializer(feat_init_options));

  // Initialize the chi squared test table with confidence level 0.95
  // https://github.com/KumarRobotics/msckf_vio/blob/050c50defa5a7fd9a04c1eed5687b405f02919b5/src/msckf_vio.cpp#L215-L221
  for (int i = 1; i < 500; i++) {
    boost::math::chi_squared chi_squared_dist(i);
    chi_squared_table[i] = boost::math::quantile(chi_squared_dist, 0.95);
  }
}

//feature_vec:追踪上大平面的小特征(未按不同大平面作分类?),以MSCKF特征为主,还补充了其它有效特征
//feature_vec_used:(传入为空,应该是要返回的结果数据) 被用作大平面初始化和估计的有效点特征. 插入此列表的点特征会被从feature_vec中删除!
//feat2plane:次新帧所有追踪上大平面的小特征,原始记录
void UpdaterPlane::init_vio_plane(std::shared_ptr<State> state, std::vector<std::shared_ptr<ov_core::Feature>> &feature_vec,
                                  std::vector<std::shared_ptr<ov_core::Feature>> &feature_vec_used,
                                  const std::map<size_t, size_t> &feat2plane) {

  // Return if no features from both msckf and slam state
  if (feature_vec.empty() && state->_features_SLAM.empty())
    return;
  if (feat2plane.empty())
    return;

  // Start timing
  boost::posix_time::ptime rT0, rT1, rT2, rT3, rT4, rT5;
  rT0 = boost::posix_time::microsec_clock::local_time();

  // 0. Get all timestamps our clones are at (and thus valid measurement times)
  std::vector<double> clonetimes;
  for (const auto &clone_imu : state->_clones_IMU) {
    clonetimes.emplace_back(clone_imu.first);
  }

  // 1. Clean all feature measurements and make sure they all have valid clone times
  std::vector<std::shared_ptr<ov_core::Feature>> feature_vec_valid; //整理待处理的小特征集合,整理结果重新存进此处
                                                                          //不在原始记录feat2plane中的小特征清除掉
                                                                          //追踪的大平面已经在滑窗中的清除掉
  auto it0 = feature_vec.begin();
  while (it0 != feature_vec.end()) {

    // Don't add if this feature is not on a plane
    if (feat2plane.find((*it0)->featid) == feat2plane.end()) {                //清除不在 原始追踪记录中的 feature_vec 特征
      it0++;
      continue;
    }

    // Skip if the plane has already been added to the state vector
    size_t planeid = feat2plane.at((*it0)->featid);
    if (state->_features_PLANE.find(planeid) != state->_features_PLANE.end()) {//skip: 追踪上的大平面已在滑窗中的大平面特征
      it0++;
      continue;
    }

    // Clean the feature
    (*it0)->clean_old_measurements(clonetimes);

    // Count how many measurements
    int ct_meas = 0;
    for (const auto &pair : (*it0)->timestamps) {
      ct_meas += (*it0)->timestamps[pair.first].size();
    }

    // Remove if we don't have enough
    if (ct_meas < 2) {  //skip观测数不足2个的小特征
      //(*it0)->to_delete = true; // NOTE: do not delete since could be incomplete track
      it0 = feature_vec.erase(it0);
    } else {
      feature_vec_valid.push_back((*it0));
      it0++;
    }
  }
  rT1 = boost::posix_time::microsec_clock::local_time();

  // 2. Create vector of cloned *CAMERA* poses at each of our clone timesteps
  // 预计算滑窗所有相机pose: { 相机id: {时间戳:相机pose} }
  std::unordered_map<size_t, std::unordered_map<double, FeatureInitializer::ClonePose>> clones_cam;
  for (const auto &clone_calib : state->_calib_IMUtoCAM) {

    // For this camera, create the vector of camera poses
    std::unordered_map<double, FeatureInitializer::ClonePose> clones_cami;
    for (const auto &clone_imu : state->_clones_IMU) {

      // Get current camera pose
      Eigen::Matrix<double, 3, 3> R_GtoCi = clone_calib.second->Rot() * clone_imu.second->Rot();
      Eigen::Matrix<double, 3, 1> p_CioinG = clone_imu.second->pos() - R_GtoCi.transpose() * clone_calib.second->pos();

      // Append to our map
      clones_cami.insert({clone_imu.first, FeatureInitializer::ClonePose(R_GtoCi, p_CioinG)});
    }

    // Append to our map
    clones_cam.insert({clone_calib.first, clones_cami});
  }

  // 3. Try to triangulate all MSCKF or new SLAM features that have measurements
  // 尝试三角化小特征点:包括msckf和slam特征
  auto it1 = feature_vec_valid.begin();
  while (it1 != feature_vec_valid.end()) {

    // Triangulate the feature and remove if it fails
    bool success_tri = true;
    if (initializer_feat->config().triangulate_1d) {
      success_tri = initializer_feat->single_triangulation_1d(*it1, clones_cam);
    } else {
      success_tri = initializer_feat->single_triangulation(*it1, clones_cam);
    }

    // Gauss-newton refine the feature
    bool success_refine = true;
    if (initializer_feat->config().refine_features) {
      success_refine = initializer_feat->single_gaussnewton(*it1, clones_cam);
    }

    // Remove the feature if not a success
    if (!success_tri || !success_refine) {
      //(*it1)->to_delete = true; // NOTE: do not delete since could be incomplete track
      it1 = feature_vec_valid.erase(it1);
      continue;
    }
    it1++;
  }
  rT2 = boost::posix_time::microsec_clock::local_time();

  // Sort based on track length, want to update with max track MSCKFs
  //按点特征自己的track次数对 feature_vec_valid 排序
  std::sort(feature_vec_valid.begin(), feature_vec_valid.end(),
            [](const std::shared_ptr<Feature> &a, const std::shared_ptr<Feature> &b) -> bool {
              size_t asize = 0;
              size_t bsize = 0;
              for (const auto &pair : a->timestamps)
                asize += pair.second.size();
              for (const auto &pair : b->timestamps)
                bsize += pair.second.size();
              return asize < bsize;
            });

  // MSCKF: Check how many features lie on the same plane!
  std::map<size_t, size_t> plane_feat_count;                            //大平面被追踪的次数.  (统计的次数超过20个就不再继续统计,以控制规模)
  std::map<size_t, std::vector<std::shared_ptr<Feature>>> plane_feats;  //大平面被追踪的具体小特征列表  // index by plane id
  std::map<size_t, std::set<double>> plane_feat_clones;                 //大平面被追踪的具体帧列表 
  for (auto &feat : feature_vec_valid) {
    if (feat2plane.find(feat->featid) == feat2plane.end())
      continue;
    size_t planeid = feat2plane.at(feat->featid);
    if (state->_features_PLANE.find(planeid) != state->_features_PLANE.end())
      continue;
    if ((int)plane_feat_count[planeid] > state->_options.max_msckf_plane)
      continue;
    plane_feat_count[planeid]++;
    plane_feats[planeid].push_back(feat);
    for (auto const &calib : feat->timestamps) {
      for (auto const &time : calib.second) {                           //点特征自己在具体单个相机下的观测帧列表
        plane_feat_clones[planeid].insert(time);
      }
    }
  }

  // SLAM: append features if they lie on a plane!
  // TODO: if we do this, the whole system seems to be a lot worst
  // TODO: how can we see if the SLAM feature is an inlier or not????
  //  for (auto &feat : state->_features_SLAM) {
  //    if (feat2plane.find(feat.first) == feat2plane.end())
  //      continue;
  //    size_t planeid = feat2plane.at(feat.first);
  //    if (state->_features_PLANE.find(planeid) != state->_features_PLANE.end())
  //      continue;
  //    plane_feat_count[planeid]++;
  //    auto featptr = std::make_shared<Feature>();
  //    featptr->featid = feat.second->_featid;
  //    featptr->p_FinG = feat.second->get_xyz(false);
  //    assert(feat.second->_feat_representation == LandmarkRepresentation::Representation::GLOBAL_3D);
  //    plane_feats[planeid].push_back(featptr);
  //  }

  // Debug print out stats
  for (auto const &planect : plane_feat_count) {
    size_t clonect = (plane_feat_clones.find(planect.first) == plane_feat_clones.end()) ? 0 : plane_feat_clones.at(planect.first).size();
    PRINT_DEBUG(BOLDCYAN "[PLANE-INIT]: plane %zu has %zu feats (%zu clones)!\n" RESET, planect.first, planect.second, clonect);
  }

  // 4. Try to initialize a guess for each plane's CP
  // For each plane we have, lets recover its CP linearization point
  // 4:到此处,追踪的大平面都是新大平面:不在滑窗中.而且还整理好了各个大平面上的追踪点.此处尝试初始化新的大平面参数cp并优化之!
  // 4-1,对单个大平面的追踪点集,用ransac线性解尝试初始化平面参数cp
  // 4-2,再利用单个大平面的观测小点,构建点的BA残差和点面距离残差,对相关的面特征参数cp,点特征坐标,相机pose(fixed),相机内参和外参(fixed)作联合优化 .
    // 成功的大平面缓存进 plane_estimates_cp_inG[plane_id],
    // 成功的大平面对应点集替换进 plane_feats[plane_id]
  std::map<size_t, Eigen::Vector3d> plane_estimates_cp_inG; //单个大平面成功优化估计的记录:  {大平面id:大平面参数cp}
                                                            //单个大平面对应的小点集记录,成功的大平面和失败的大平面都有(成功大平面只保留内点集). {大平面id:大平面对应的小点集}
  for (const auto &featspair : plane_feats) {   //featspair:单个大平面所有的追踪点集

    // Initial guess of the plane
    //线性解平面参数,作为平面初始化值
    //输入单个大平面所有追踪点集,随机采样5点拟合平面:内点距离阈值是5厘米
      //有效平面要求内点数至少10个内点 && 内点数占比至少0.8    
    Eigen::Vector4d abcd;
    if (!PlaneFitting::plane_fitting(plane_feats[featspair.first], abcd, state->_options.plane_init_min_feat,
                                     state->_options.plane_init_max_cond))
      continue;
    double avg_error_tri = 0.0;
    for (const auto &feat : featspair.second)
      avg_error_tri += std::abs(PlaneFitting::point_to_plane_distance(feat->p_FinG, abcd));
    avg_error_tri /= (double)featspair.second.size();

    // Print stats of feature before we try to optimize them...
    if (state->_options.use_groundtruths && !state->_true_planes.empty() && !state->_true_features.empty()) {
      double featd_avg_norm = 0.0;
      Eigen::Vector3d featd_avg = Eigen::Vector3d::Zero();
      for (auto const &featpair : featspair.second) {
        Eigen::Vector3d featd = featpair->p_FinG - state->_true_features.at(featpair->featid);
        featd_avg += featd;
        featd_avg_norm += featd.norm();
      }
      featd_avg /= (double)featspair.second.size();
      featd_avg_norm /= (double)featspair.second.size();
      PRINT_INFO(YELLOW "[PLANE-GT]: feat avg | feat_diff = %.3f, %.3f,%.3f | BEFORE\n" RESET, featd_avg(0), featd_avg(1), featd_avg(2));
      PRINT_INFO(YELLOW "[PLANE-GT]: feat avg | feat_diff_norm = %.3f | BEFORE\n" RESET, featd_avg_norm);
    }

    // Try to optimize this plane and features together
    // TODO: be smarter about how we get focal length here...
    double focal_length = state->_cam_intrinsics_cameras.at(0)->get_value()(0);
    double sigma_px_norm = _options.sigma_pix / focal_length;
    double sigma_c = state->_options.sigma_constraint;
    Eigen::Vector3d cp_inG = -abcd.head(3) * abcd(3); //cp_inG=平面法向n*截距d:  法向方向的逆向.模长是截距d. 平面方程是n*p-d=0
    Eigen::VectorXd stateI = state->_imu->pose()->value();
    Eigen::VectorXd calib0 = state->_calib_IMUtoCAM.at(0)->value();
        
    // 初始化单个大平面参数cp后,利用单个大平面的观测小点,构建点的BA残差和点面距离残差,对相关的面特征参数cp,点特征坐标,相机pose(fixed),相机内参和外参(fixed)作联合优化    
    // 注意这里只有单个大平面
    // plane_feats[featspair.first]:单个大平面的追踪点集
    // cp_inG:单个大平面的参数
    // clones_cam:所有帧的相机pose
    if (!PlaneFitting::optimize_plane(plane_feats[featspair.first], cp_inG, clones_cam, sigma_px_norm, sigma_c, false, stateI, calib0))
      continue;
    //优化成功
    abcd.head(3) = cp_inG / cp_inG.norm();
    abcd(3) = -cp_inG.norm();
    double avg_error_opt = 0.0;
    for (const auto &feat : featspair.second)
      avg_error_opt += std::abs(PlaneFitting::point_to_plane_distance(feat->p_FinG, abcd));
    avg_error_opt /= (double)featspair.second.size();

    // Set groundtruth if we have it and can set it
    if (state->_options.use_groundtruths && !state->_true_planes.empty() && !state->_true_features.empty()) {
      Eigen::Vector3d cpdiff = cp_inG - state->_true_planes.at(featspair.first);
      PRINT_INFO(YELLOW "[PLANE-GT]: plane %zu | cp_diff = %.3f, %.3f,%.3f |\n" RESET, featspair.first, cpdiff(0), cpdiff(1), cpdiff(2));
      cp_inG = state->_true_planes.at(featspair.first);
      double featd_avg_norm = 0.0;
      Eigen::Vector3d featd_avg = Eigen::Vector3d::Zero();
      for (auto const &featpair : featspair.second) {
        Eigen::Vector3d featd = featpair->p_FinG - state->_true_features.at(featpair->featid);
        featd_avg += featd;
        featd_avg_norm += featd.norm();
        PRINT_INFO(YELLOW "[PLANE-GT]: feat %zu | feat_diff = %.3f, %.3f,%.3f |\n" RESET, featpair->featid, featd(0), featd(1), featd(2));
        featpair->p_FinG = state->_true_features.at(featpair->featid);
      }
      featd_avg /= (double)featspair.second.size();
      featd_avg_norm /= (double)featspair.second.size();
      PRINT_INFO(YELLOW "[PLANE-GT]: feat avg | feat_diff = %.3f, %.3f,%.3f |\n" RESET, featd_avg(0), featd_avg(1), featd_avg(2));
      PRINT_INFO(YELLOW "[PLANE-GT]: feat avg | feat_diff_norm = %.3f |\n" RESET, featd_avg_norm);
    }

    // Success! Lets add the plane!
    plane_estimates_cp_inG.insert({featspair.first, cp_inG});   //将单个大平面的成功估计结果保存下来
    PRINT_INFO(BOLDCYAN "[PLANE-INIT]: plane %zu | %.3f err tri | %.3f err opt |\n" RESET, featspair.first, avg_error_tri, avg_error_opt);
  }
  rT3 = boost::posix_time::microsec_clock::local_time();

  // 5. use the features that are on the same plane to initialize tha plane
  // 用最小二乘优化得到面特征参数后,现将面特征作为新的状态扩充滑窗状态:状态扩充,协方差扩充.
  for (auto const &planepair : plane_estimates_cp_inG) {  //planepair:单个成功估计的大平面.{大平面id:大平面参数cp}

    // Get all features / variables for this plane
    size_t planeid = planepair.first;
    Eigen::Vector3d cp_inG = planepair.second;
    std::vector<std::shared_ptr<Feature>> features = plane_feats.at(planeid); //features:单个成功估计的大平面对应小点的内点集合
    assert(features.size() >= 3);
    assert(state->_features_PLANE.find(planeid) == state->_features_PLANE.end());

    // Calculate the max possible measurement size (1 constraint for each feat time)
    size_t max_meas_size = 0;
    for (size_t i = 0; i < features.size(); i++) {
      for (const auto &pair : features.at(i)->timestamps) {
        max_meas_size += 3 * features.at(i)->timestamps[pair.first].size();
      }
      if (features.at(i)->timestamps.empty()) {
        max_meas_size += 1; // slam feature has constraint measurement
      }
    }
    size_t max_hx_size = state->max_covariance_size();

    // Large Jacobian and residual of *all* features for this update
    Eigen::VectorXd res_big = Eigen::VectorXd::Zero(max_meas_size);
    Eigen::MatrixXd Hx_big = Eigen::MatrixXd::Zero(max_meas_size, max_hx_size);
    Eigen::MatrixXd Hcp_big = Eigen::MatrixXd::Zero(max_meas_size, 3);
    std::unordered_map<std::shared_ptr<Type>, size_t> Hx_mapping;
    std::vector<std::shared_ptr<Type>> Hx_order_big;
    size_t ct_jacob = 0;
    size_t ct_meas = 0;

    // Compute linear system for each feature, nullspace project, and reject
    //对单个点特征,区分slam点和msckf点,构建总残差res:点BA和点面残差;对滑窗状态雅可比Hx,对点特征雅可比Hf,对面特征雅可比Hcp
    //1)目标:
      // res:点BA和点面残差
      // Hx:
          // slam点:对pose状态和slam点自己的雅可比
          // msckf点:对pose状态的雅可比,对点特征自己的雅可比会被边缘化掉
      // Hcp:对面特征雅可比
    //2)求解步骤:
      //直接求res,Hx,Hf,Hcp:不区分slam点和msckf点
      //对slam点,将Hf丟进Hx中.(slam点已经是状态x):   剩余 res, Hx, Hcp
      //对msckf点,求Hf左零空间,边缘化res,Hf,Hx,Hcp: 剩余 res, Hx, Hcp
    
    //3)将所有点的res, Hx, Hcp叠加起来
    for (auto const &feature : features) {//features:单个大平面的追踪点集

      // If we are a SLAM feature, then we should append the feature Jacobian to Hx
      // Otherwise, if we are a MSCKF feature, then we should nullspace project
      // Thus there will be two set of logics below depending on this flag!
      // NOTE: this does not work yet for when we have an aruco tag feature....
      bool is_slam_feature = (state->_features_SLAM.find(feature->featid) != state->_features_SLAM.end());
      assert((int)feature->featid >= state->_options.max_aruco_features);

      // Convert our feature into our current format
      UpdaterHelper::UpdaterHelperFeature feat;
      feat.featid = feature->featid;
      feat.uvs = feature->uvs;
      feat.uvs_norm = feature->uvs_norm;
      feat.timestamps = feature->timestamps;

      // Append plane info from triangulation
      feat.planeid = planeid;
      feat.cp_FinG = cp_inG;
      feat.cp_FinG_fej = cp_inG;

      // If we are using single inverse depth, then it is equivalent to using the msckf inverse depth
      feat.feat_representation = (is_slam_feature) ? state->_options.feat_rep_slam : state->_options.feat_rep_msckf;
      if (feat.feat_representation == LandmarkRepresentation::Representation::ANCHORED_INVERSE_DEPTH_SINGLE) {
        feat.feat_representation = LandmarkRepresentation::Representation::ANCHORED_MSCKF_INVERSE_DEPTH;
      }

      // Save the position and its fej value
      assert(!LandmarkRepresentation::is_relative_representation(feat.feat_representation));
      if (LandmarkRepresentation::is_relative_representation(feat.feat_representation)) {
        feat.anchor_cam_id = feature->anchor_cam_id;
        feat.anchor_clone_timestamp = feature->anchor_clone_timestamp;
        if (is_slam_feature) {
          feat.p_FinA = state->_features_SLAM.at(feature->featid)->get_xyz(false);
          feat.p_FinA_fej = state->_features_SLAM.at(feature->featid)->get_xyz(true);
        } else {
          feat.p_FinA = feature->p_FinA;
          feat.p_FinA_fej = feature->p_FinA;
        }
      } else {
        if (is_slam_feature) {
          feat.p_FinG = state->_features_SLAM.at(feature->featid)->get_xyz(false);
          feat.p_FinG_fej = state->_features_SLAM.at(feature->featid)->get_xyz(true);
        } else {
          feat.p_FinG = feature->p_FinG;
          feat.p_FinG_fej = feature->p_FinG;
        }
      }

      // Our return values (feature jacobian, state jacobian, residual, and order of state jacobian)
      Eigen::MatrixXd H_f;
      Eigen::MatrixXd H_x;
      Eigen::VectorXd res;
      std::vector<std::shared_ptr<Type>> Hx_order;

      // Get the Jacobian for this feature
      // 1)等效于点特征的jacobian_full计算
      // 2)额外,如果点特征和大平面有关联,就还考虑点面残差:
      //    2-1)点面残差, 叠加到 res 末尾
      //    2-2)对点特征雅可比, 叠加到 H_x末尾
      //    2-3)对面特征雅可比H_c_plane, 
      //        2-3-1)如果面特征在滑窗 _features_PLANE 中, 面特征作为状态叠加到H_x末尾
      //        2-3-1)如果面特征不在滑窗中(新面特征), 面特征作为特征叠加到H_f末尾      
      double sigma_c = state->_options.const_init_multi * state->_options.sigma_constraint;
      UpdaterHelper::get_feature_jacobian_full(state, feat, _options.sigma_pix, sigma_c, H_f, H_x, res, Hx_order);

      //拆分H_f, 取出单点特征所有观测,对,面特征雅可比H_cp
      //拆分H_f, 取出单点特征所有观测,对,点特征雅可比H_f
      // Separate our the derivative in respect to the plane
      assert(H_f.cols() == 6); // TODO: handle single depth
      Eigen::MatrixXd H_cp = H_f.block(0, H_f.cols() - 3, H_f.rows(), 3);
      H_f = H_f.block(0, 0, H_f.rows(), 3).eval();

      // Append to Hx if SLAM feature, else nullspace project (if this is a MSCKF feature)
      //如果点特征是slam特征(即已在滑窗状态中),对点特征雅可比H_f需要叠加进H_x中
      //如果点特征是msckf特征,对点特征雅可比H_f需要被边缘化掉:H_f左零置为0,其余的res,H_x,H_cp左零边缘化
      if (is_slam_feature) {
        Eigen::MatrixXd H_xf = H_x;
        H_xf.conservativeResize(H_x.rows(), H_x.cols() + H_f.cols());
        H_xf.block(0, H_x.cols(), H_x.rows(), H_f.cols()) = H_f;
        std::vector<std::shared_ptr<Type>> Hxf_order = Hx_order;
        Hxf_order.push_back(state->_features_SLAM.at(feature->featid));
        H_x = H_xf;
        Hx_order = Hxf_order;
      } else {
        UpdaterPlane::nullspace_project_inplace(H_f, H_x, H_cp, res);
      }

      // We are good!!! Append to our large H vector
      size_t ct_hx = 0;
      for (const auto &var : Hx_order) {

        // Ensure that this variable is in our Jacobian
        if (Hx_mapping.find(var) == Hx_mapping.end()) {
          Hx_mapping.insert({var, ct_jacob});
          Hx_order_big.push_back(var);
          ct_jacob += var->size();
        }

        // Append to our large Jacobian
        Hx_big.block(ct_meas, Hx_mapping[var], H_x.rows(), var->size()) = H_x.block(0, ct_hx, H_x.rows(), var->size());
        ct_hx += var->size();
      }
      Hcp_big.block(ct_meas, 0, H_cp.rows(), H_cp.cols()) = H_cp;

      // Append our residual and move forward
      res_big.block(ct_meas, 0, res.rows(), 1) = res;
      ct_meas += res.rows();
    }

    // Now we have stacked all features, resize to the smaller amount
    assert(ct_meas > 3);
    assert(ct_meas <= max_meas_size);
    assert(ct_jacob <= max_hx_size);
    res_big.conservativeResize(ct_meas, 1);
    Hx_big.conservativeResize(ct_meas, ct_jacob);
    Hcp_big.conservativeResize(ct_meas, 3);

    // Perform measurement compression to reduce update size
    //QR分解Hcp, 可以压缩观测方程res, Hx, Hcp
    UpdaterPlane::measurement_compress_inplace(Hx_big, Hcp_big, res_big);
    assert(Hx_big.rows() > 0);
    Eigen::MatrixXd R_big = Eigen::MatrixXd::Identity(res_big.rows(), res_big.rows());

    // Create plane feature pointer
    auto plane = std::make_shared<Vec>(3);
    plane->set_value(cp_inG);
    plane->set_fej(cp_inG);

    // Try to initialize (internally checks chi2)
    //1)对新增的单个大平面特征,扩展滑窗状态和协方差
    //2)然后基于单个大平面特征所有观测(左零方程),来作ESKF后验刷新,刷新滑窗所有状态和协方差        
    //3)将新的特征加入滑窗特征列表:state->_features_PLANE    
    if (StateHelper::initialize(state, plane, Hx_order_big, Hx_big, Hcp_big, R_big, res_big, state->_options.const_init_chi2)) {

      // Append to the state vector
      state->_features_PLANE.insert({planeid, plane});
      PRINT_INFO(GREEN "[PLANE-INIT]: plane %d inited | cp_init = %.3f,%.3f,%.3f | cp = %.3f,%.3f,%.3f |\n" RESET, planeid, cp_inG(0),
                 cp_inG(1), cp_inG(2), plane->value()(0), plane->value()(1), plane->value()(2));

      // Get what the marginal covariance init'ed was...
      Eigen::MatrixXd cov_marg = StateHelper::get_marginal_covariance(state, {plane});
      Eigen::Vector3d sigmas = cov_marg.diagonal().transpose().cwiseSqrt();
      PRINT_INFO(GREEN "[PLANE-INIT]: plane prior = %.3f, %.3f, %.3f | inflation = %.3f |\n" RESET, sigmas(0), sigmas(1), sigmas(2),
                 state->_options.const_init_multi);

      // Remove all features from the MSCKF vector if we updated with it
      std::set<size_t> ids;
      for (auto const &feature : features) { //单个大平面的追踪点集
        assert(ids.find(feature->featid) == ids.end());
        ids.insert(feature->featid);
        feature_vec_used.push_back(feature);
      }
      it0 = feature_vec.begin();
      while (it0 != feature_vec.end()) {
        if (ids.find((*it0)->featid) != ids.end()) {
          (*it0)->to_delete = true;                 //单个msckf点特征已经用作平面初始化了,此处删除此msckf点特征
          feature_vec_used.push_back((*it0));
          it0 = feature_vec.erase(it0);
        } else {
          it0++;
        }
      }
    } else {
      PRINT_INFO(RED "[PLANE-INIT]: plane %d init failed | cp = %.3f,%.3f,%.3f |\n" RESET, planeid, cp_inG(0), cp_inG(1), cp_inG(2),
                 plane->value()(0));
    }
  }
}

void UpdaterPlane::nullspace_project_inplace(Eigen::MatrixXd &H_f, Eigen::MatrixXd &H_x, Eigen::MatrixXd &H_cp, Eigen::VectorXd &res) {

  // Make sure we have enough measurements to project
  assert(H_f.rows() >= H_f.cols());

  // Apply the left nullspace of H_f to all variables
  // Based on "Matrix Computations 4th Edition by Golub and Van Loan"
  // See page 252, Algorithm 5.2.4 for how these two loops work
  // They use "matlab" index notation, thus we need to subtract 1 from all index
  Eigen::JacobiRotation<double> tempHo_GR;
  for (int n = 0; n < H_f.cols(); ++n) {
    for (int m = (int)H_f.rows() - 1; m > n; m--) {
      // Givens matrix G
      tempHo_GR.makeGivens(H_f(m - 1, n), H_f(m, n));
      // Multiply G to the corresponding lines (m-1,m) in each matrix
      // Note: we only apply G to the nonzero cols [n:Ho.cols()-n-1], while
      //       it is equivalent to applying G to the entire cols [0:Ho.cols()-1].
      (H_f.block(m - 1, n, 2, H_f.cols() - n)).applyOnTheLeft(0, 1, tempHo_GR.adjoint());
      (H_x.block(m - 1, 0, 2, H_x.cols())).applyOnTheLeft(0, 1, tempHo_GR.adjoint());
      (H_cp.block(m - 1, 0, 2, H_cp.cols())).applyOnTheLeft(0, 1, tempHo_GR.adjoint());
      (res.block(m - 1, 0, 2, 1)).applyOnTheLeft(0, 1, tempHo_GR.adjoint());
    }
  }

  // The H_f jacobian max rank is 3 if it is a 3d position, thus size of the left nullspace is Hf.rows()-3
  // NOTE: need to eigen3 eval here since this experiences aliasing!
  // H_f = H_f.block(H_f.cols(),0,H_f.rows()-H_f.cols(),H_f.cols()).eval();
  H_x = H_x.block(H_f.cols(), 0, H_x.rows() - H_f.cols(), H_x.cols()).eval();
  H_cp = H_cp.block(H_f.cols(), 0, H_cp.rows() - H_f.cols(), H_cp.cols()).eval();
  res = res.block(H_f.cols(), 0, res.rows() - H_f.cols(), res.cols()).eval();

  // Sanity check
  assert(H_x.rows() == res.rows());
  assert(H_cp.rows() == res.rows());
}

void UpdaterPlane::measurement_compress_inplace(Eigen::MatrixXd &H_x, Eigen::MatrixXd &H_cp, Eigen::VectorXd &res) {

  // Return if H_x is a fat matrix (there is no need to compress in this case)
  if (H_x.rows() <= H_x.cols())
    return;

  // Do measurement compression through givens rotations
  // Based on "Matrix Computations 4th Edition by Golub and Van Loan"
  // See page 252, Algorithm 5.2.4 for how these two loops work
  // They use "matlab" index notation, thus we need to subtract 1 from all index
  Eigen::JacobiRotation<double> tempHo_GR;
  for (int n = 0; n < H_x.cols(); n++) {
    for (int m = (int)H_x.rows() - 1; m > n; m--) {
      // Givens matrix G
      tempHo_GR.makeGivens(H_x(m - 1, n), H_x(m, n));
      // Multiply G to the corresponding lines (m-1,m) in each matrix
      // Note: we only apply G to the nonzero cols [n:Ho.cols()-n-1], while
      //       it is equivalent to applying G to the entire cols [0:Ho.cols()-1].
      (H_x.block(m - 1, n, 2, H_x.cols() - n)).applyOnTheLeft(0, 1, tempHo_GR.adjoint());
      (H_cp.block(m - 1, 0, 2, H_cp.cols())).applyOnTheLeft(0, 1, tempHo_GR.adjoint());
      (res.block(m - 1, 0, 2, 1)).applyOnTheLeft(0, 1, tempHo_GR.adjoint());
    }
  }

  // If H is a fat matrix, then use the rows
  // Else it should be same size as our state
  int r = std::min(H_x.rows(), H_x.cols());

  // Construct the smaller jacobian and residual after measurement compression
  assert(r <= H_x.rows());
  H_x.conservativeResize(r, H_x.cols());
  H_cp.conservativeResize(r, H_cp.cols());
  res.conservativeResize(r, res.cols());
}
