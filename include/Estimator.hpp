#pragma once

#include <vector>
#include <iostream>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include "types.hpp"
#include "so3.hpp"


namespace RigVis
{
	/*
		Robocentric error-state Kalman filter.

		The filter's reference frame "W" is NOT a fixed world frame: it is the
		IMU body frame at the previous camera frame. After every camera frame the
		estimated motion (R, p) is handed out and the reference moves to the new
		body frame (R = I, p = 0 again, velocity and gravity rotated along).

		Why: the absolute position and heading are unobservable for VO + IMU.
		A filter that keeps them in its state lets their uncertainty grow without
		bound, and the linearized updates then start "correcting" them with
		information that is not there (the classic EKF-VIO inconsistency: on the
		full KITTI run the pose jumped by up to 0.3 m per frame and the path was
		5 % too long). In the robocentric form those directions are simply not in
		the state; the absolute pose is the chain of the fused frame motions.

		Error state (18): δx = [δtheta, δposition, δvelocity, δbiasGyro, δbiasAcc, δgravity]
		Gravity is a state because "down" in the moving reference frame changes;
		the accelerometer keeps it observable (this is what bounds roll/pitch).
	*/
	using CovarianceMatrix = Eigen::Matrix<double, 18, 18>;
	using TransitionMatrix = Eigen::Matrix<double, 18, 18>;
	using NoiseMappingMatrix = Eigen::Matrix<double, 18, 12>;
	using NoiseCovarianceMatrix = Eigen::Matrix<double, 12, 12>;
	using Matrix6d = Eigen::Matrix<double, 6, 6>;

	struct FilterState
	{
		// Timestamp of the state currently stored here.
		double timestamp_s = -1.0;

		// Body / IMU position and velocity, expressed in the reference frame W
		// (= body frame at the previous camera frame, see above).
		Eigen::Vector3d positionW =
			Eigen::Vector3d::Zero();

		Eigen::Vector3d velocityW =
			Eigen::Vector3d::Zero();

		// Rotation that converts a body-frame vector into reference coordinates.
		Eigen::Quaterniond quaternionWB =
			Eigen::Quaterniond::Identity();

		Eigen::Vector3d biasAcc =
			Eigen::Vector3d::Zero();

		Eigen::Vector3d biasGyro =
			Eigen::Vector3d::Zero();

		// Gravity vector in the reference frame (|g| = 9.81).
		Eigen::Vector3d gravityW =
			Eigen::Vector3d(0.0, 0.0, -9.81);
	};

	struct ImuNoise
	{
		// Continuous-time densities (sigma per sqrt(Hz)). Initial values, tune later.
		// Tuned on the full KITTI 0027 run with the 100 Hz extract data.
		double gyroMeasurementStd = 0.005;
		double accelMeasurementStd = 0.20;  // vibration; use ~0.5 with the 10 Hz sync data

		double gyroBiasWalkStd = 0.0001;
		double accelBiasWalkStd = 0.001;

		// IMU dropouts: the KITTI 100 Hz OXTS data has gaps of up to 1.9 s. Over
		// a gap the integration holds one stale reading (a turn in between is
		// missed), so the prediction is marked unreliable: the covariance is
		// inflated and the next visual update skips the chi^2 gate.
		double maxGapSeconds = 0.25;
		double gapRotationStdRad = 0.2;     // added per gap (~11 deg)
		double gapVelocityStd = 1.0;        // m/s
	};

	// Result of one measurement update, for logging.
	struct UpdateResult
	{
		bool accepted = false;
		double mahalanobis2 = 0.0;          // r^T S^-1 r, chi^2 with 6 dof if the noise model is right
		Eigen::Matrix<double, 6, 1> residual = Eigen::Matrix<double, 6, 1>::Zero();  // [rot rad, pos m]
	};

	class Estimator
	{
		public:
			Estimator();

			// Starts the filter with the velocity and gravity expressed in the
			// CURRENT body frame. Until this is called, IMU samples only update the
			// "previous sample" (nothing is integrated).
			void initialize(double timestamp_s, const Eigen::Vector3d& velocityBody,
							const Eigen::Vector3d& gravityBody);
			bool initialized() const { return stateInitialized_; }

			void imuPropagate(const ImuSample& imuInput);

			// Integrates from the state time to timestamp_s with the latest IMU sample
			// (zero-order hold). Called at a camera timestamp so that the state and
			// the visual measurement refer to the same instant.
			void propagateTo(double timestamp_s);

			// Visual measurement of the body (IMU) motion since the previous camera
			// frame: deltaRotation = R_WB, deltaPosition = p_W (W = previous body frame).
			// noise: 6x6 covariance of [rotation (rad); position (m)].
			// Measurements failing the chi^2 gate are rejected (state unchanged).
			UpdateResult updateRelativePose(const Eigen::Quaterniond& deltaRotation,
											const Eigen::Vector3d& deltaPosition, const Matrix6d& noise,
											double chiSquareGate = 22.46);  // 6 dof, 99.9 %

			// Ends the current camera interval: returns the estimated body motion
			// since the previous camera frame and moves the reference frame to the
			// current body frame. Call once per camera frame, after the update.
			Eigen::Isometry3d shiftReferenceFrame();

			const CovarianceMatrix& covariance() const { return covariance_; }
			const FilterState& state() const { return nominalState_; }
			void setImuNoise(const ImuNoise& noise) { imuNoise_ = noise; }
			const ImuNoise& imuNoise() const { return imuNoise_; }
			std::size_t imuGaps() const { return imuGaps_; }



		private:

			// One IMU integration step over dt with raw (bias-including) readings.
			void integrate(const Eigen::Vector3d& gyroMeasured, const Eigen::Vector3d& accMeasured, double dt);

			void propagateCovariance(
				const Eigen::Vector3d& gyroBody,
				const Eigen::Vector3d& accBody,
				double dt);

			// Inflates P after an IMU gap (see ImuNoise::maxGapSeconds).
			void handleImuGap(double gapSeconds);

			bool imuInitiated_{false};
			bool stateInitialized_{false};
			bool skipGateOnce_{false};
			int imuIndex_{0};
			std::size_t imuGaps_{0};

			// Add a state factor it is used everywhere
			FilterState nominalState_;
			ImuNoise imuNoise_;
			ImuSample prevImuInput_;

			CovarianceMatrix covariance_ =
				CovarianceMatrix::Identity() * 1e-3;


	};


} // namespace RigVis
