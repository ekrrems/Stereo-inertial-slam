#include "LocalBundleAdjuster.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <unordered_map>
#include <unordered_set>

#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <Eigen/Cholesky>
#include <Eigen/Geometry>

#include "StereoFrontend.hpp"  // Keyframe, SparseMapPoint, Observation

namespace RigVis
{
	namespace
	{
		struct Intrinsics
		{
			double fx, fy, cx, cy, focalTimesBaseline;
		};

		// ---------------------------------------------------------------- residuals
		//
		// One residual block = one observation of point P (world) in keyframe T_cw.
		// Parameter blocks: quaternion q_cw (4, Eigen order x y z w), translation t_cw (3), point (3).
		// Returns (observed - predicted) / sigma, so squared norm = v'Pv.
		// Ceres differentiates this automatically (the Jacobian J = A).

		struct StereoReprojectionError
		{
			StereoReprojectionError(double uLeft, double v, double uRight, double sigma, const Intrinsics& k)
				: uLeft_(uLeft), v_(v), uRight_(uRight), inverseSigma_(1.0 / sigma), k_(k) {}

			template <typename T>
			bool operator()(const T* const q, const T* const t, const T* const point, T* residual) const
			{
				const Eigen::Map<const Eigen::Quaternion<T>> rotation(q);
				const Eigen::Map<const Eigen::Matrix<T, 3, 1>> translation(t);
				const Eigen::Map<const Eigen::Matrix<T, 3, 1>> pointWorld(point);

				const Eigen::Matrix<T, 3, 1> pointCamera = rotation * pointWorld + translation;  // P_c = R P_w + t
				if (pointCamera.z() <= T(1e-6)) return false;                                   // behind camera

				const T inverseDepth = T(1.0) / pointCamera.z();
				const T uLeft = T(k_.fx) * pointCamera.x() * inverseDepth + T(k_.cx);
				const T v = T(k_.fy) * pointCamera.y() * inverseDepth + T(k_.cy);
				const T uRight = uLeft - T(k_.focalTimesBaseline) * inverseDepth;

				residual[0] = (T(uLeft_) - uLeft) * T(inverseSigma_);
				residual[1] = (T(v_) - v) * T(inverseSigma_);
				residual[2] = (T(uRight_) - uRight) * T(inverseSigma_);
				return true;
			}

			static ceres::CostFunction* create(double uLeft, double v, double uRight, double sigma, const Intrinsics& k)
			{
				return new ceres::AutoDiffCostFunction<StereoReprojectionError, 3, 4, 3, 3>(
					new StereoReprojectionError(uLeft, v, uRight, sigma, k));
			}

			double uLeft_, v_, uRight_, inverseSigma_;
			Intrinsics k_;
		};

		struct MonoReprojectionError
		{
			MonoReprojectionError(double uLeft, double v, double sigma, const Intrinsics& k)
				: uLeft_(uLeft), v_(v), inverseSigma_(1.0 / sigma), k_(k) {}

			template <typename T>
			bool operator()(const T* const q, const T* const t, const T* const point, T* residual) const
			{
				const Eigen::Map<const Eigen::Quaternion<T>> rotation(q);
				const Eigen::Map<const Eigen::Matrix<T, 3, 1>> translation(t);
				const Eigen::Map<const Eigen::Matrix<T, 3, 1>> pointWorld(point);

				const Eigen::Matrix<T, 3, 1> pointCamera = rotation * pointWorld + translation;
				if (pointCamera.z() <= T(1e-6)) return false;

				const T inverseDepth = T(1.0) / pointCamera.z();
				residual[0] = (T(uLeft_) - (T(k_.fx) * pointCamera.x() * inverseDepth + T(k_.cx))) * T(inverseSigma_);
				residual[1] = (T(v_) - (T(k_.fy) * pointCamera.y() * inverseDepth + T(k_.cy))) * T(inverseSigma_);
				return true;
			}

			static ceres::CostFunction* create(double uLeft, double v, double sigma, const Intrinsics& k)
			{
				return new ceres::AutoDiffCostFunction<MonoReprojectionError, 2, 4, 3, 3>(
					new MonoReprojectionError(uLeft, v, sigma, k));
			}

			double uLeft_, v_, inverseSigma_;
			Intrinsics k_;
		};

		// Relative pose between two keyframes (dense ICP), poses as T_cw (camera-from-world):
		//   T_A<-B = T_cA * T_cB^-1,   E = T_A<-B * T_meas^-1,   e = (Log R_E, t_E)
		//   residual = L e  with  L^T L = information  ->  ||residual||^2 = e^T Info e
		class RelativePoseFactor
		{
		public:
			RelativePoseFactor(const Eigen::Isometry3d& aFromB, const Eigen::Matrix<double, 6, 6>& information)
				: measuredInverse_(aFromB.inverse())
			{
				sqrtInformation_ = information.llt().matrixU();
			}

			template <typename T>
			bool operator()(const T* const qA, const T* const tA, const T* const qB, const T* const tB, T* residual) const
			{
				const Eigen::Map<const Eigen::Quaternion<T>> rotationA(qA), rotationB(qB);
				const Eigen::Map<const Eigen::Matrix<T, 3, 1>> translationA(tA), translationB(tB);
				// T_cB^-1
				const Eigen::Quaternion<T> rotationBInv = rotationB.conjugate();
				const Eigen::Matrix<T, 3, 1> translationBInv = -(rotationBInv * translationB);
				// T_A<-B
				const Eigen::Quaternion<T> rotationAB = rotationA * rotationBInv;
				const Eigen::Matrix<T, 3, 1> translationAB = rotationA * translationBInv + translationA;
				// E = T_A<-B * T_meas^-1
				const Eigen::Quaternion<T> qm(measuredInverse_.linear().template cast<T>());
				const Eigen::Matrix<T, 3, 1> tm = measuredInverse_.translation().template cast<T>();
				const Eigen::Quaternion<T> rotationE = rotationAB * qm;
				const Eigen::Matrix<T, 3, 1> translationE = rotationAB * tm + translationAB;

				const T quaternionWxyz[4] = {rotationE.w(), rotationE.x(), rotationE.y(), rotationE.z()};
				Eigen::Matrix<T, 6, 1> error;
				ceres::QuaternionToAngleAxis(quaternionWxyz, error.data());
				error.template tail<3>() = translationE;
				Eigen::Map<Eigen::Matrix<T, 6, 1>> out(residual);
				out = sqrtInformation_.template cast<T>() * error;
				return true;
			}

			static ceres::CostFunction* create(const Eigen::Isometry3d& aFromB, const Eigen::Matrix<double, 6, 6>& information)
			{
				return new ceres::AutoDiffCostFunction<RelativePoseFactor, 6, 4, 3, 4, 3>(
					new RelativePoseFactor(aFromB, information));
			}

		private:
			Eigen::Isometry3d measuredInverse_;
			Eigen::Matrix<double, 6, 6> sqrtInformation_;
		};

		// Bookkeeping for one residual block, needed for the outlier test.
		struct ResidualInfo
		{
			ceres::ResidualBlockId id = nullptr;
			std::size_t pointIndex = 0;
			std::size_t observationIndex = 0;
			bool stereo = false;
		};

		double rmsNormalizedResidual(ceres::Problem& problem)
		{
			ceres::Problem::EvaluateOptions evaluateOptions;
			evaluateOptions.apply_loss_function = false;  // plain v'Pv, not robustified
			std::vector<double> residuals;
			double cost = 0.0;
			problem.Evaluate(evaluateOptions, &cost, &residuals, nullptr, nullptr);
			if (residuals.empty()) return 0.0;
			double sum = 0.0;
			for (const double r : residuals) sum += r * r;
			return std::sqrt(sum / static_cast<double>(residuals.size()));
		}
	}

	BundleAdjustmentSummary runLocalBundleAdjustment(std::vector<Keyframe>& keyframes,
													 std::vector<SparseMapPoint>& mapPoints,
													 const StereoCalibration& calibration,
													 const BundleAdjustmentOptions& options,
													 const std::vector<RelativePoseConstraint>* relativeConstraints)
	{
		const auto startTime = std::chrono::steady_clock::now();
		BundleAdjustmentSummary summary;
		if (keyframes.size() < 2 || options.windowKeyframes == 0) return summary;

		const Intrinsics intrinsics{calibration.left.fx, calibration.left.fy, calibration.left.cx,
									calibration.left.cy, calibration.left.fx * calibration.baseline_m};

		// ---- 1. Free keyframes: the newest window, never keyframe 0 (world frame).
		const std::size_t firstFree =
			keyframes.size() > options.windowKeyframes ? keyframes.size() - options.windowKeyframes : 1;
		auto isFree = [&](std::size_t keyframeIndex) { return keyframeIndex >= firstFree; };

		// ---- 2. Points: valid, >= minObservations, seen by at least one free keyframe.
		// Collected through the free keyframes' own point lists (mapPointIds), so the
		// cost depends on the window size, not on the size of the whole map.
		std::vector<std::size_t> pointIndices;
		std::unordered_set<std::size_t> collected;
		std::unordered_set<std::size_t> fixedKeyframes;
		for (std::size_t k = firstFree; k < keyframes.size(); ++k) {
			for (const int id : keyframes[k].mapPointIds) {
				if (id < 0) continue;
				const std::size_t p = static_cast<std::size_t>(id);
				if (p >= mapPoints.size() || !collected.insert(p).second) continue;  // already seen via another keyframe

				const SparseMapPoint& point = mapPoints[p];
				if (!point.valid || point.observations.size() < options.minObservationsPerPoint) continue;
				pointIndices.push_back(p);

				// ---- 3. Fixed keyframes: also see these points but are outside the window.
				for (const Observation& observation : point.observations) {
					if (observation.keyframeId < keyframes.size() && !isFree(observation.keyframeId)) {
						fixedKeyframes.insert(observation.keyframeId);
					}
				}
			}
		}
		if (pointIndices.empty()) return summary;

		// ---- Parameter storage. Poses as T_cw (camera-from-world): projection needs it.
		std::unordered_map<std::size_t, std::size_t> poseSlot;  // keyframe index -> slot
		std::vector<std::array<double, 4>> quaternions;        // x y z w (Eigen order)
		std::vector<std::array<double, 3>> translations;
		auto addPose = [&](std::size_t keyframeIndex) {
			if (poseSlot.count(keyframeIndex)) return;
			const Eigen::Isometry3d cameraFromWorld = keyframes[keyframeIndex].worldFromCamera.inverse();
			const Eigen::Quaterniond q(cameraFromWorld.linear());
			poseSlot[keyframeIndex] = quaternions.size();
			quaternions.push_back({q.x(), q.y(), q.z(), q.w()});
			translations.push_back({cameraFromWorld.translation().x(), cameraFromWorld.translation().y(),
									cameraFromWorld.translation().z()});
		};
		for (std::size_t k = firstFree; k < keyframes.size(); ++k) addPose(k);
		for (const std::size_t k : fixedKeyframes) addPose(k);

		// ICP edges that touch the window. An endpoint outside the window joins as
		// a fixed pose. Added now, before any pointer into the pose arrays is taken.
		std::vector<const RelativePoseConstraint*> usedConstraints;
		if (relativeConstraints) {
			for (const RelativePoseConstraint& c : *relativeConstraints) {
				if (c.keyframeA >= keyframes.size() || c.keyframeB >= keyframes.size()) continue;
				if (!isFree(c.keyframeA) && !isFree(c.keyframeB)) continue;
				for (const std::size_t k : {c.keyframeA, c.keyframeB}) {
					if (!isFree(k) && !poseSlot.count(k)) { fixedKeyframes.insert(k); addPose(k); }
				}
				usedConstraints.push_back(&c);
			}
		}

		std::vector<std::array<double, 3>> points(pointIndices.size());
		for (std::size_t i = 0; i < pointIndices.size(); ++i) {
			const Eigen::Vector3d& position = mapPoints[pointIndices[i]].positionWorld;
			points[i] = {position.x(), position.y(), position.z()};
		}

		// ---- 4. Build the problem: one residual block per observation.
		ceres::Problem problem;
		std::vector<ResidualInfo> residualInfos;
		const double huberStereo = std::sqrt(options.chi2Stereo);
		const double huberMono = std::sqrt(options.chi2Mono);

		for (std::size_t i = 0; i < pointIndices.size(); ++i) {
			const SparseMapPoint& point = mapPoints[pointIndices[i]];
			for (std::size_t o = 0; o < point.observations.size(); ++o) {
				const Observation& observation = point.observations[o];
				if (observation.keyframeId >= keyframes.size()) continue;
				const std::size_t slot = poseSlot.at(observation.keyframeId);

				// Skip observations that start behind the camera (would fail the first evaluation).
				const Eigen::Vector3d pointCamera =
					Eigen::Quaterniond(quaternions[slot][3], quaternions[slot][0], quaternions[slot][1], quaternions[slot][2]) *
						point.positionWorld +
					Eigen::Vector3d(translations[slot][0], translations[slot][1], translations[slot][2]);
				if (pointCamera.z() <= 1e-6) continue;

				const double sigma = std::pow(1.2, observation.octave);
				ResidualInfo info;
				info.pointIndex = pointIndices[i];
				info.observationIndex = o;
				info.stereo = observation.isStereo();

				// Every point is CREATED from stereo, but later keyframes re-observe it
				// through the LEFT image (tracking). If that keyframe's own stereo
				// matching found no right partner, uRight = -1: a mono observation.
				// It still constrains pose and point via (u_L, v), just not depth via
				// u_R. Feeding it to the stereo residual would compare -1 against a
				// predicted u_R of ~600 px: a fake 600 px error.
				if (info.stereo) {
					info.id = problem.AddResidualBlock(
						StereoReprojectionError::create(observation.pixel.x(), observation.pixel.y(),
														observation.uRight, sigma, intrinsics),
						new ceres::HuberLoss(huberStereo),
						quaternions[slot].data(), translations[slot].data(), points[i].data());
				} else {
					info.id = problem.AddResidualBlock(
						MonoReprojectionError::create(observation.pixel.x(), observation.pixel.y(), sigma, intrinsics),
						new ceres::HuberLoss(huberMono),
						quaternions[slot].data(), translations[slot].data(), points[i].data());
				}
				residualInfos.push_back(info);
			}
		}
		if (residualInfos.empty()) return summary;

		const double huberRelative = std::sqrt(options.chi2Relative);
		std::vector<ceres::ResidualBlockId> relativeIds;
		for (const RelativePoseConstraint* c : usedConstraints) {
			const std::size_t a = poseSlot.at(c->keyframeA), b = poseSlot.at(c->keyframeB);
			relativeIds.push_back(problem.AddResidualBlock(RelativePoseFactor::create(c->aFromB, c->information),
														   new ceres::HuberLoss(huberRelative), quaternions[a].data(),
														   translations[a].data(), quaternions[b].data(),
														   translations[b].data()));
		}
		summary.relativeConstraints = relativeIds.size();

		// Quaternions live on a manifold: 4 numbers, 3 degrees of freedom, stays unit length.
		for (auto& [keyframeIndex, slot] : poseSlot) {
			if (problem.HasParameterBlock(quaternions[slot].data())) {
				problem.SetManifold(quaternions[slot].data(), new ceres::EigenQuaternionManifold);
			}
		}

		// Datum: fixed keyframes keep their poses. If none exist (e.g. early in the
		// run), fix the oldest free keyframe so the solution cannot float away.
		std::size_t fixedCount = 0;
		for (const std::size_t k : fixedKeyframes) {
			const std::size_t slot = poseSlot.at(k);
			if (!problem.HasParameterBlock(quaternions[slot].data())) continue;
			problem.SetParameterBlockConstant(quaternions[slot].data());
			problem.SetParameterBlockConstant(translations[slot].data());
			++fixedCount;
		}
		if (fixedCount == 0) {
			const std::size_t slot = poseSlot.at(firstFree);
			if (problem.HasParameterBlock(quaternions[slot].data())) {
				problem.SetParameterBlockConstant(quaternions[slot].data());
				problem.SetParameterBlockConstant(translations[slot].data());
				fixedCount = 1;
			}
		}

		summary.freeKeyframes = (keyframes.size() - firstFree) - (fixedKeyframes.empty() ? 1 : 0);
		summary.fixedKeyframes = fixedCount;
		summary.points = pointIndices.size();
		summary.observations = residualInfos.size();
		summary.rmsNormalizedBefore = rmsNormalizedResidual(problem);

		ceres::Solver::Options solverOptions;
		// Eliminate points, then solve the pose system: dense for a small local
		// window, sparse for large windows (global BA after a loop closure).
		solverOptions.linear_solver_type =
			(keyframes.size() - firstFree) > 40 ? ceres::SPARSE_SCHUR : ceres::DENSE_SCHUR;
		solverOptions.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
		solverOptions.num_threads = options.numThreads;
		solverOptions.minimizer_progress_to_stdout = false;
		solverOptions.logging_type = ceres::SILENT;

		// ---- 5. First pass (robust), then outlier test on the plain residuals.
		ceres::Solver::Summary ceresSummary;
		solverOptions.max_num_iterations = options.iterationsFirstPass;
		ceres::Solve(solverOptions, &problem, &ceresSummary);
		summary.initialCost = ceresSummary.initial_cost;
		summary.iterations = static_cast<int>(ceresSummary.iterations.size());

		std::vector<const ResidualInfo*> outliers;
		for (const ResidualInfo& info : residualInfos) {
			double cost = 0.0;
			problem.EvaluateResidualBlock(info.id, /*apply_loss_function=*/false, &cost, nullptr, nullptr);
			const double chi2 = 2.0 * cost;  // Ceres cost = 0.5 * ||r||^2, r already divided by sigma
			if (chi2 > (info.stereo ? options.chi2Stereo : options.chi2Mono)) outliers.push_back(&info);
		}
		for (const ResidualInfo* info : outliers) problem.RemoveResidualBlock(info->id);
		summary.outliersRemoved = outliers.size();
		for (const ceres::ResidualBlockId id : relativeIds) {   // same test for the ICP edges
			double cost = 0.0;
			problem.EvaluateResidualBlock(id, false, &cost, nullptr, nullptr);
			if (2.0 * cost > options.chi2Relative) { problem.RemoveResidualBlock(id); ++summary.relativeOutliers; }
		}

		// ---- 6. Second pass without the outliers.
		solverOptions.max_num_iterations = options.iterationsSecondPass;
		ceres::Solve(solverOptions, &problem, &ceresSummary);
		summary.finalCost = ceresSummary.final_cost;
		summary.iterations += static_cast<int>(ceresSummary.iterations.size());
		summary.rmsNormalizedAfter = rmsNormalizedResidual(problem);

		// ---- 7. Write back: free poses (T_cw -> T_wc) and all adjusted points.
		std::vector<Eigen::Isometry3d> oldWorldFromCamera;
		oldWorldFromCamera.reserve(keyframes.size() - firstFree);
		for (std::size_t k = firstFree; k < keyframes.size(); ++k) {
			oldWorldFromCamera.push_back(keyframes[k].worldFromCamera);
		}
		for (std::size_t k = firstFree; k < keyframes.size(); ++k) {
			const std::size_t slot = poseSlot.at(k);
			Eigen::Isometry3d cameraFromWorld = Eigen::Isometry3d::Identity();
			cameraFromWorld.linear() = Eigen::Quaterniond(quaternions[slot][3], quaternions[slot][0],
														  quaternions[slot][1], quaternions[slot][2])
										   .normalized()
										   .toRotationMatrix();
			cameraFromWorld.translation() = Eigen::Vector3d(translations[slot][0], translations[slot][1],
															translations[slot][2]);
			keyframes[k].worldFromCamera = cameraFromWorld.inverse();
		}
		std::vector<char> optimized(mapPoints.size(), 0);
		for (std::size_t i = 0; i < pointIndices.size(); ++i) {
			mapPoints[pointIndices[i]].positionWorld = Eigen::Vector3d(points[i][0], points[i][1], points[i][2]);
			optimized[pointIndices[i]] = 1;
		}

		// Points seen by a free keyframe but NOT in the problem (single observation,
		// no redundancy) would be left behind when their keyframe moves. Move them
		// rigidly with it: P_new = T_wc_new * T_wc_old^-1 * P_old. Then they still
		// reproject exactly where they were measured, and tracking stays consistent.
		std::vector<char> moved(mapPoints.size(), 0);
		for (std::size_t k = firstFree; k < keyframes.size(); ++k) {
			const Eigen::Isometry3d correction = keyframes[k].worldFromCamera * oldWorldFromCamera[k - firstFree].inverse();
			for (const int pointIndex : keyframes[k].mapPointIds) {
				if (pointIndex < 0) continue;
				const std::size_t p = static_cast<std::size_t>(pointIndex);
				if (p >= mapPoints.size() || optimized[p] || moved[p] || !mapPoints[p].valid) continue;
				mapPoints[p].positionWorld = correction * mapPoints[p].positionWorld;
				moved[p] = 1;
			}
		}

		// ---- 8. Drop outlier observations from the map and from the keyframes,
		// so tracking stops matching against them. Erase from the back so the
		// remaining observation indices stay valid.
		std::sort(outliers.begin(), outliers.end(), [](const ResidualInfo* a, const ResidualInfo* b) {
			return a->pointIndex != b->pointIndex ? a->pointIndex < b->pointIndex
												  : a->observationIndex > b->observationIndex;
		});
		for (const ResidualInfo* info : outliers) {
			SparseMapPoint& point = mapPoints[info->pointIndex];
			const Observation observation = point.observations[info->observationIndex];
			std::vector<int>& ids = keyframes[observation.keyframeId].mapPointIds;
			if (observation.keypointIndex >= 0 && static_cast<std::size_t>(observation.keypointIndex) < ids.size() &&
				ids[static_cast<std::size_t>(observation.keypointIndex)] == static_cast<int>(info->pointIndex)) {
				ids[static_cast<std::size_t>(observation.keypointIndex)] = -1;
			}
			point.observations.erase(point.observations.begin() + static_cast<long>(info->observationIndex));
			if (point.observations.empty()) point.valid = false;
		}

		summary.ran = true;
		summary.milliseconds = std::chrono::duration<double, std::milli>(
			std::chrono::steady_clock::now() - startTime).count();
		return summary;
	}
}
