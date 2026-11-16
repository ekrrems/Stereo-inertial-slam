#pragma once

#include <vector>
#include <iostream>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include "types.hpp"
#include "so3.hpp"


namespace RigVis
{
	using CovarianceMatrix = Eigen::Matrix<double, 15, 15>;
	using TransitionMatrix = Eigen::Matrix<double, 15, 15>;
	using NoiseMappingMatrix = Eigen::Matrix<double, 15, 12>;
	using NoiseCovarianceMatrix = Eigen::Matrix<double, 12, 12>;

	struct FilterState
	{
		// Timestamp of the state currently stored here.
		double timestamp_s = -1.0;

		// Body / IMU position and velocity, expressed in world coordinates.
		Eigen::Vector3d positionW =
			Eigen::Vector3d::Zero();

		Eigen::Vector3d velocityW =
			Eigen::Vector3d::Zero();

		// Rotation that converts a body-frame vector into world coordinates.
		Eigen::Quaterniond quaternionWB =
			Eigen::Quaterniond::Identity();

		Eigen::Vector3d biasAcc =
			Eigen::Vector3d::Zero();

		Eigen::Vector3d biasGyro =
			Eigen::Vector3d::Zero();
	};

	struct ImuNoise
	{
		// Initial placeholder values. Tune these later from KITTI sensor specs.
		double gyroMeasurementStd = 0.01;
		double accelMeasurementStd = 0.10;

		double gyroBiasWalkStd = 0.0001;
		double accelBiasWalkStd = 0.001;
	};

	class Estimator
	{
		public:
			Estimator();

			void imuPropagate(const ImuSample& imuInput);

			const CovarianceMatrix& covariance() const { return covariance_; }
			const FilterState& state() const { return nominalState_; }



		private:

			void propagateCovariance(
				const Eigen::Vector3d& gyroBody,
				const Eigen::Vector3d& accBody,
				double dt);

			bool imuInitiated_{false};
			int imuIndex_{0};

			// Add a state factor it is used everywhere
			FilterState nominalState_;
			ImuNoise imuNoise_;
			ImuSample prevImuInput_;

			CovarianceMatrix covariance_ =
				CovarianceMatrix::Identity() * 1e-3;


			// Gravity (NEU coordinate system)
			Eigen::Vector3d gravityW_ = Eigen::Vector3d(0.0, 0.0, -9.81);


	};

	
} // namespace RigVis
