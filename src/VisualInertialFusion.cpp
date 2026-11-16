#include "VisualInertialFusion.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <iostream>
#include <sstream>
#include <vector>

namespace RigVis
{
	namespace
	{
		constexpr double radToDeg = 57.29577951308232;
		constexpr double gravityMagnitude = 9.81;
	}

	VisualInertialFusion::VisualInertialFusion(const StereoCalibration& calibration,
											   double initializationWindowSeconds)
		: cameraFromImu_(calibration.camera_from_imu),
		  imuFromCamera_(calibration.camera_from_imu.inverse()),
		  windowSeconds_(initializationWindowSeconds)
	{
		if (!calibration.has_imu_extrinsics) {
			std::cout << "VIO: no IMU-camera calibration found, using identity (results will be wrong)\n";
		}
	}

	void VisualInertialFusion::addImu(const ImuSample& sample)
	{
		estimator_.imuPropagate(sample);
		if (!estimator_.initialized()) {
			windowImu_.push_back({sample.timestamp_s, sample.acceleration, sample.angularVel});
			while (windowImu_.size() > 1 && windowImu_.front().timestamp_s < sample.timestamp_s - 2.0 * windowSeconds_) {
				windowImu_.pop_front();
			}
		}
	}

	void VisualInertialFusion::addCameraFrame(double timestamp_s, const Eigen::Isometry3d& worldFromCamera,
											  bool visualPoseValid)
	{
		lastVisualPose_ = worldFromCamera;
		log_ = FusionFrameLog{};

		if (!estimator_.initialized()) {
			if (visualPoseValid) {
				windowFrames_.push_back({timestamp_s, worldFromCamera * cameraFromImu_});
				tryInitialize();
			} else {
				windowFrames_.clear();  // tracking broke: the window must be continuous
			}
			previousWorldFromCamera_ = worldFromCamera;
			havePreviousVisual_ = visualPoseValid;
			log_.initialized = estimator_.initialized();
			return;
		}

		// State and measurement must refer to the same instant.
		estimator_.propagateTo(timestamp_s);

		if (visualPoseValid && havePreviousVisual_) {
			// Camera motion -> body (IMU) motion:  T_Bc_Bk = T_BC * T_Cc_Ck * T_CB
			const Eigen::Isometry3d deltaCamera = previousWorldFromCamera_.inverse() * worldFromCamera;
			const Eigen::Isometry3d deltaBody = imuFromCamera_ * deltaCamera * cameraFromImu_;

			Matrix6d noise = Matrix6d::Zero();
			noise.topLeftCorner<3, 3>().diagonal().setConstant(visualNoise_.rotationStdRad * visualNoise_.rotationStdRad);
			noise.bottomRightCorner<3, 3>().diagonal().setConstant(visualNoise_.positionStdM * visualNoise_.positionStdM);

			const UpdateResult update = estimator_.updateRelativePose(Eigen::Quaterniond(deltaBody.linear()),
																	  deltaBody.translation(), noise);
			log_.updated = update.accepted;
			log_.rejected = !update.accepted;
			log_.mahalanobis2 = update.mahalanobis2;
			log_.residualRotationDeg = update.residual.head<3>().norm() * radToDeg;
			log_.residualPositionM = update.residual.tail<3>().norm();
			if (update.accepted) ++accepted_;
			else ++rejected_;
		}
		// Hand out the fused motion of this interval and move the filter's
		// reference frame to the current body frame (robocentric filter).
		worldFromBody_ = worldFromBody_ * estimator_.shiftReferenceFrame();
		previousWorldFromCamera_ = worldFromCamera;
		havePreviousVisual_ = visualPoseValid;

		const FilterState& state = estimator_.state();
		log_.initialized = true;
		log_.speedMps = state.velocityW.norm();
		log_.biasGyroDegPerS = state.biasGyro * radToDeg;
		log_.biasAcc = state.biasAcc;
	}

	// Gravity + velocity from the first seconds of VO and IMU (see the header).
	void VisualInertialFusion::tryInitialize()
	{
		if (windowFrames_.size() < 8) return;
		const WindowFrame& first = windowFrames_.front();
		const WindowFrame& last = windowFrames_.back();
		const double t0 = first.timestamp_s;
		if (last.timestamp_s - t0 < windowSeconds_) return;
		if (windowImu_.size() < 2 || windowImu_.back().timestamp_s < last.timestamp_s - 0.2) {
			return;  // IMU does not reach the end of the window (yet)
		}
		if (windowImu_.front().timestamp_s > t0) {
			windowFrames_.pop_front();  // IMU starts after this frame: start the window later
			return;
		}

		// ---- 1. Integrate the IMU through the window, starting at the VO rotation R_0.
		// For each frame k: S_k = double integral of R f (position part without
		// v0 and g), and the rotation reached by the gyro. Piecewise constant
		// readings: the mid-point of two consecutive samples.
		const std::size_t frameCount = windowFrames_.size();
		std::vector<Eigen::Vector3d> S(frameCount, Eigen::Vector3d::Zero());
		std::vector<Eigen::Vector3d> V(frameCount, Eigen::Vector3d::Zero());  // single integral of R f
		Eigen::Matrix3d R = first.worldFromBody.linear();
		Eigen::Vector3d integratedVelocity = Eigen::Vector3d::Zero();
		Eigen::Vector3d integratedPosition = Eigen::Vector3d::Zero();
		Eigen::Vector3d meanRotatedForce = Eigen::Vector3d::Zero();
		double meanGyroNorm = 0.0;
		double time = t0;
		std::size_t frame = 1;
		int forceSamples = 0;

		// The last frame may come a few ms before the IMU sample after it: the
		// final piece holds the last reading (end time = infinity).
		for (std::size_t i = 1; i <= windowImu_.size() && frame < frameCount; ++i) {
			const bool hold = i == windowImu_.size();
			const WindowImu& a = windowImu_[i - 1];
			const WindowImu& b = hold ? windowImu_[i - 1] : windowImu_[i];
			const double pieceEnd = hold ? std::numeric_limits<double>::infinity() : b.timestamp_s;
			if (pieceEnd <= time) continue;
			const Eigen::Vector3d force = 0.5 * (a.acceleration + b.acceleration);
			const Eigen::Vector3d gyro = 0.5 * (a.angularVel + b.angularVel);
			// This piece covers (a.t, b.t]; split it at frame timestamps.
			while (time < pieceEnd && frame < frameCount) {
				const double stepEnd = std::min(pieceEnd, windowFrames_[frame].timestamp_s);
				const double dt = stepEnd - time;
				const Eigen::Vector3d rotatedForce = R * force;
				integratedPosition += integratedVelocity * dt + 0.5 * rotatedForce * dt * dt;
				integratedVelocity += rotatedForce * dt;
				R = R * so3Exp(gyro * dt).toRotationMatrix();
				time = stepEnd;
				if (time >= windowFrames_[frame].timestamp_s) {
					S[frame] = integratedPosition;
					V[frame] = integratedVelocity;
					++frame;
				}
			}
			meanRotatedForce += R * force;
			meanGyroNorm += gyro.norm();
			++forceSamples;
		}
		if (frame < frameCount || forceSamples < 5) return;
		meanRotatedForce /= forceSamples;
		meanGyroNorm /= forceSamples;

		const double displacement = (last.worldFromBody.translation() - first.worldFromBody.translation()).norm();
		Eigen::Vector3d gravity, velocityStart;
		double fitRms = 0.0;
		const char* mode = "";

		if (displacement < 0.05 && meanGyroNorm < 0.01) {
			// ---- 2a. Static start: the accelerometer sees only gravity.
			gravity = -meanRotatedForce;
			velocityStart.setZero();
			mode = "static";
		} else {
			// ---- 2b. Moving start: least squares for x = [v0; g] from
			//   (p_k - p_0) - S_k = v0 t_k + 1/2 g t_k^2
			const int rows = static_cast<int>(3 * (frameCount - 1));
			Eigen::MatrixXd A = Eigen::MatrixXd::Zero(rows, 6);
			Eigen::VectorXd y(rows);
			for (std::size_t k = 1; k < frameCount; ++k) {
				const double tk = windowFrames_[k].timestamp_s - t0;
				const int r = static_cast<int>(3 * (k - 1));
				A.block<3, 3>(r, 0) = Eigen::Matrix3d::Identity() * tk;
				A.block<3, 3>(r, 3) = Eigen::Matrix3d::Identity() * 0.5 * tk * tk;
				y.segment<3>(r) =
					windowFrames_[k].worldFromBody.translation() - first.worldFromBody.translation() - S[k];
			}
			const Eigen::VectorXd x = A.colPivHouseholderQr().solve(y);
			gravity = x.tail<3>();

			const double magnitude = gravity.norm();
			if (std::abs(magnitude - gravityMagnitude) > 0.3) {
				std::cout << "VIO init: |g| = " << magnitude << " m/s^2, retrying with a later window\n";
				windowFrames_.pop_front();
				return;
			}
			// ---- 3. Fix |g| = 9.81 (direction kept) and re-solve v0 alone.
			gravity *= gravityMagnitude / magnitude;
			Eigen::VectorXd yv = y - A.rightCols<3>() * gravity;
			velocityStart = A.leftCols<3>().colPivHouseholderQr().solve(yv);
			fitRms = std::sqrt((A.leftCols<3>() * velocityStart - yv).squaredNorm() / (frameCount - 1));
			if (fitRms > 0.10) {
				// Positions not explained by IMU + constant g: bad VO stretch or timing.
				std::cout << "VIO init: alignment residual " << fitRms << " m, retrying with a later window\n";
				windowFrames_.pop_front();
				return;
			}
			mode = "moving";
		}
		if (mode[0] == 's') gravity *= gravityMagnitude / gravity.norm();

		// Velocity at the last frame: v_end = v0 + g t + integral of R f
		const double duration = last.timestamp_s - t0;
		const Eigen::Vector3d velocity = velocityStart + gravity * duration + V[frameCount - 1];
		// The filter works in the body frame: rotate velocity and gravity into it.
		const Eigen::Matrix3d bodyFromWorld = last.worldFromBody.linear().transpose();
		estimator_.initialize(last.timestamp_s, bodyFromWorld * velocity, bodyFromWorld * gravity);
		worldFromBody_ = last.worldFromBody;

		// Gravity in camera axes of the first frame: y is down, so expect ~ (0, +9.8, 0).
		std::cout << std::fixed << std::setprecision(3) << "VIO initialized (" << mode << " start, "
				  << frameCount << " frames, fit rms " << fitRms << " m) at t = " << last.timestamp_s
				  << " s | g in first-camera frame: " << gravity.transpose() << " | speed " << velocity.norm()
				  << " m/s\n" << std::defaultfloat;
		windowFrames_.clear();
		windowImu_.clear();
	}

	Eigen::Isometry3d VisualInertialFusion::cameraPose() const
	{
		if (!estimator_.initialized()) return lastVisualPose_;
		return worldFromBody_ * imuFromCamera_;
	}

	std::string fusionCsvHeader()
	{
		return "vio_initialized,vio_updated,vio_rejected,vio_mahalanobis2,vio_res_rot_deg,vio_res_pos_m,"
			   "vio_speed_mps,bg_x_dps,bg_y_dps,bg_z_dps,ba_x,ba_y,ba_z";
	}

	std::string fusionCsvRow(const FusionFrameLog& log)
	{
		std::ostringstream row;
		row << std::setprecision(6) << log.initialized << ',' << log.updated << ',' << log.rejected << ','
			<< log.mahalanobis2 << ',' << log.residualRotationDeg << ',' << log.residualPositionM << ','
			<< log.speedMps << ',' << log.biasGyroDegPerS.x() << ',' << log.biasGyroDegPerS.y() << ','
			<< log.biasGyroDegPerS.z() << ',' << log.biasAcc.x() << ',' << log.biasAcc.y() << ',' << log.biasAcc.z();
		return row.str();
	}
}
