/*
	This script propagates the state using the IMU input
*/

#include "Estimator.hpp"


namespace RigVis
{
	Estimator::Estimator()
	{
		std::cout << "Estimator is initialized\n";
	}

	void Estimator::imuPropagate(const ImuSample& imuInput)
	{
		if (!imuInitiated_ || nominalState_.timestamp_s < 0.0)
		{
			prevImuInput_ = imuInput;
			nominalState_.timestamp_s = imuInput.timestamp_s;

			imuInitiated_ = true;
			imuIndex_++;
			return;
		}

		// Calculate the delta time
		const double dt = imuInput.timestamp_s - prevImuInput_.timestamp_s;

		if (dt <= 0.0)
		{
			// error message
			// throw std::runtime_error(
			std::cout <<
				"IMU timeline gap has an issue: dt = " <<
				std::to_string(dt) << " seconds, index: " << std::to_string(imuIndex_) << "\n";
			imuIndex_++;
			return;
		}

		// correct the measured IMU readings
		const Eigen::Vector3d accBody = imuInput.acceleration - nominalState_.biasAcc;
		const Eigen::Vector3d gyroBody = imuInput.angularVel - nominalState_.biasGyro;

		// Task: Make it middle dt

		// Convert body frame to the world frame
		const Eigen::Vector3d accWorld = nominalState_.quaternionWB * accBody + gravityW_;

		// Update the Nominal State
		nominalState_.positionW += nominalState_.velocityW * dt + 0.5 * accWorld * dt * dt;
		nominalState_.velocityW += accWorld * dt;

		// Update the Quaternions
		const Eigen::Vector3d delta_theta = gyroBody * dt;
		const double angle_rad = delta_theta.norm();

		Eigen::Quaterniond delta_rotation = Eigen::Quaterniond::Identity();

		if (angle_rad > 1e-12) {
			delta_rotation = Eigen::Quaterniond(
				Eigen::AngleAxisd(
					angle_rad,
					delta_theta / angle_rad));
		}

		nominalState_.quaternionWB =(nominalState_.quaternionWB * delta_rotation).normalized();
		propagateCovariance(gyroBody, accBody, dt);

		// Update the timestamps
		nominalState_.timestamp_s = imuInput.timestamp_s;
		prevImuInput_ = imuInput;
		imuIndex_++;
	}
	// covariance structure for stochastic implementation
	void Estimator::propagateCovariance(
		const Eigen::Vector3d& gyroBody,
		const Eigen::Vector3d& accBody,
		const double dt)
	{
		// Error-state order:
		// δx = [δtheta, δposition, δvelocity, δbiasGyro, δbiasAcc]
		constexpr int thetaIndex = 0;
		constexpr int positionIndex = 3;
		constexpr int velocityIndex = 6;
		constexpr int gyroBiasIndex = 9;
		constexpr int accBiasIndex = 12;

		const Eigen::Matrix3d identity3 =
			Eigen::Matrix3d::Identity();

		// Orientation at the beginning of this IMU interval.
		const Eigen::Matrix3d rotationWB =
			nominalState_.quaternionWB.toRotationMatrix();

		// δx_dot = F δx + G n
		TransitionMatrix F =
			TransitionMatrix::Zero();

		// δtheta_dot = -skew(gyroBody) δtheta - δbiasGyro - gyroNoise
		F.block<3, 3>(thetaIndex, thetaIndex) =
			-skew(gyroBody);

		F.block<3, 3>(thetaIndex, gyroBiasIndex) =
			-identity3;

		// δposition_dot = δvelocity
		F.block<3, 3>(positionIndex, velocityIndex) =
			identity3;

		// δvelocity_dot =
		// -R * skew(accBody) * δtheta
		// -R * δbiasAcc
		// -R * accelNoise
		F.block<3, 3>(velocityIndex, thetaIndex) =
			-rotationWB * skew(accBody);

		F.block<3, 3>(velocityIndex, accBiasIndex) =
			-rotationWB;

		// Noise order:
		// n = [gyroNoise, accelNoise, gyroBiasWalk, accelBiasWalk]
		NoiseMappingMatrix G =
			NoiseMappingMatrix::Zero();

		G.block<3, 3>(thetaIndex, 0) =
			-identity3;

		G.block<3, 3>(velocityIndex, 3) =
			-rotationWB;

		G.block<3, 3>(gyroBiasIndex, 6) =
			identity3;

		G.block<3, 3>(accBiasIndex, 9) =
			identity3;

		// Continuous-time noise covariance Q.
		NoiseCovarianceMatrix Q =
			NoiseCovarianceMatrix::Zero();

		Q.block<3, 3>(0, 0) =
			identity3 *
			imuNoise_.gyroMeasurementStd *
			imuNoise_.gyroMeasurementStd;

		Q.block<3, 3>(3, 3) =
			identity3 *
			imuNoise_.accelMeasurementStd *
			imuNoise_.accelMeasurementStd;

		Q.block<3, 3>(6, 6) =
			identity3 *
			imuNoise_.gyroBiasWalkStd *
			imuNoise_.gyroBiasWalkStd;

		Q.block<3, 3>(9, 9) =
			identity3 *
			imuNoise_.accelBiasWalkStd *
			imuNoise_.accelBiasWalkStd;

		// Discretize the local error system over dt.
		const TransitionMatrix Phi =
			TransitionMatrix::Identity() + F * dt;

		const CovarianceMatrix discreteNoise =
			G * Q * G.transpose() * dt;

		covariance_ =
			Phi * covariance_ * Phi.transpose() +
			discreteNoise;

		// Floating-point arithmetic can cause tiny asymmetry.
		covariance_ =
			0.5 * (covariance_ + covariance_.transpose());
	}

	// Class for taking the input then using that one for updating the state

	// Add everything needed for the Error state kalman filter
}