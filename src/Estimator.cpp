/*
	Robocentric error-state Kalman filter (ESKF): IMU propagation + VO motion updates.

	Reference frame W = IMU body frame at the previous camera frame (see Estimator.hpp).
	Nominal state  x = [p, v, q, b_g, b_a, g]          (propagated with the IMU, non-linear)
	Error state   δx = [δθ, δp, δv, δb_g, δb_a, δg]    (small, Gaussian, has the covariance P)
	True state       = nominal ⊕ error:  q_true = q ⊗ Exp(δθ)  (local / right perturbation)

	Prediction (every IMU sample):  nominal integrated, P = Φ P Φ^T + G Q G^T dt
	Update (every camera frame):    z = VO motion since the previous frame = (q, p) directly,
	                                K = P H^T (H P H^T + R)^-1, δx = K r, inject, δx = 0
	Shift (every camera frame):     hand out (q, p), new reference = current body frame
*/

#include "Estimator.hpp"

#include <algorithm>


namespace RigVis
{
	namespace
	{
		// Error-state order:
		// δx = [δtheta, δposition, δvelocity, δbiasGyro, δbiasAcc, δgravity]
		constexpr int thetaIndex = 0;
		constexpr int positionIndex = 3;
		constexpr int velocityIndex = 6;
		constexpr int gyroBiasIndex = 9;
		constexpr int accBiasIndex = 12;
		constexpr int gravityIndex = 15;
		constexpr double gravityMagnitude = 9.81;
	}

	Estimator::Estimator()
	{
		std::cout << "Estimator is initialized\n";
	}

	void Estimator::initialize(double timestamp_s, const Eigen::Vector3d& velocityBody,
							   const Eigen::Vector3d& gravityBody)
	{
		nominalState_ = FilterState{};  // R = I, p = 0: the reference frame is the current body frame
		nominalState_.timestamp_s = timestamp_s;
		nominalState_.velocityW = velocityBody;
		nominalState_.gravityW = gravityBody.normalized() * gravityMagnitude;

		// Initial uncertainty (1 sigma): 0.1 m/s, 0.3 deg/s, 0.1 m/s^2,
		// gravity direction 0.5 deg (9.81 * 0.0087 = 0.085 m/s^2).
		// The pose part is exactly zero: the reference frame IS the current pose.
		const auto square = [](double value) { return value * value; };
		covariance_.setZero();
		covariance_.block<3, 3>(velocityIndex, velocityIndex).diagonal().setConstant(square(0.1));
		covariance_.block<3, 3>(gyroBiasIndex, gyroBiasIndex).diagonal().setConstant(square(0.3 * EIGEN_PI / 180.0));
		covariance_.block<3, 3>(accBiasIndex, accBiasIndex).diagonal().setConstant(square(0.1));
		covariance_.block<3, 3>(gravityIndex, gravityIndex).diagonal().setConstant(square(0.085));

		stateInitialized_ = true;
	}

	void Estimator::imuPropagate(const ImuSample& imuInput)
	{
		if (!imuInitiated_ || !stateInitialized_)
		{
			// Before initialize(): only remember the sample, so the first
			// interval after initialization has a measurement.
			prevImuInput_ = imuInput;
			imuInitiated_ = true;
			imuIndex_++;
			return;
		}

		// Delta time from the STATE time (which may be a camera timestamp after
		// propagateTo), not from the previous IMU sample.
		const double dt = imuInput.timestamp_s - nominalState_.timestamp_s;

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

		// Mid-point rule: the average of the readings at both ends of the interval
		// (the reading at t_k alone lags by dt/2: 50 ms with the 10 Hz sync data).
		integrate(0.5 * (prevImuInput_.angularVel + imuInput.angularVel),
				  0.5 * (prevImuInput_.acceleration + imuInput.acceleration), dt);

		const double sampleGap = imuInput.timestamp_s - prevImuInput_.timestamp_s;
		if (sampleGap > imuNoise_.maxGapSeconds) handleImuGap(sampleGap);

		// Update the timestamps
		nominalState_.timestamp_s = imuInput.timestamp_s;
		prevImuInput_ = imuInput;
		imuIndex_++;
	}

	void Estimator::propagateTo(double timestamp_s)
	{
		if (!stateInitialized_ || !imuInitiated_) return;
		const double dt = timestamp_s - nominalState_.timestamp_s;
		if (dt <= 0.0) return;
		integrate(prevImuInput_.angularVel, prevImuInput_.acceleration, dt);  // hold the last reading
		nominalState_.timestamp_s = timestamp_s;

		// No IMU sample for a while: this interval was dead reckoning on a stale reading.
		if (timestamp_s - prevImuInput_.timestamp_s > imuNoise_.maxGapSeconds) handleImuGap(dt);
	}

	void Estimator::handleImuGap(double gapSeconds)
	{
		++imuGaps_;
		const double scale = std::min(1.0, gapSeconds / imuNoise_.maxGapSeconds);  // at least one full step
		const double rotationVariance = imuNoise_.gapRotationStdRad * imuNoise_.gapRotationStdRad * scale;
		const double velocityVariance = imuNoise_.gapVelocityStd * imuNoise_.gapVelocityStd * scale;
		covariance_.block<3, 3>(thetaIndex, thetaIndex).diagonal().array() += rotationVariance;
		covariance_.block<3, 3>(velocityIndex, velocityIndex).diagonal().array() += velocityVariance;
		covariance_.block<3, 3>(positionIndex, positionIndex).diagonal().array() +=
			velocityVariance * gapSeconds * gapSeconds;
		skipGateOnce_ = true;
	}

	void Estimator::integrate(const Eigen::Vector3d& gyroMeasured, const Eigen::Vector3d& accMeasured, double dt)
	{
		// correct the measured IMU readings
		const Eigen::Vector3d accBody = accMeasured - nominalState_.biasAcc;
		const Eigen::Vector3d gyroBody = gyroMeasured - nominalState_.biasGyro;

		// Covariance first: F uses the orientation at the BEGINNING of the interval.
		propagateCovariance(gyroBody, accBody, dt);

		// Convert body frame to the world frame
		const Eigen::Vector3d accWorld = nominalState_.quaternionWB * accBody + nominalState_.gravityW;

		// Update the Nominal State
		nominalState_.positionW += nominalState_.velocityW * dt + 0.5 * accWorld * dt * dt;
		nominalState_.velocityW += accWorld * dt;

		// Update the Quaternions: q <- q ⊗ Exp(ω dt)
		nominalState_.quaternionWB = (nominalState_.quaternionWB * so3Exp(gyroBody * dt)).normalized();
	}

	// covariance structure for stochastic implementation
	void Estimator::propagateCovariance(
		const Eigen::Vector3d& gyroBody,
		const Eigen::Vector3d& accBody,
		const double dt)
	{
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

		// + δgravity (gravity is a state; constant, so δg_dot = 0)
		F.block<3, 3>(velocityIndex, gravityIndex) =
			identity3;

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

	/*
		VO motion measurement. In the robocentric frame the measured quantities
		ARE the pose states: z = (q_WB, p_W) of the current body in the previous
		body frame. So
		  residual r = [ Log(q̂^-1 ⊗ q_meas) ; p_meas - p̂ ]  ≈  H δx,   H = [I 0 0...; 0 I 0...]
		Velocity, biases and gravity are observed only through the correlations
		in P that the IMU propagation built up during the interval.
	*/
	UpdateResult Estimator::updateRelativePose(const Eigen::Quaterniond& deltaRotation,
											   const Eigen::Vector3d& deltaPosition, const Matrix6d& noise,
											   double chiSquareGate)
	{
		UpdateResult result;
		if (!stateInitialized_) return result;

		Eigen::Matrix<double, 6, 1> residual;
		residual.head<3>() = so3Log(nominalState_.quaternionWB.conjugate() * deltaRotation);
		residual.tail<3>() = deltaPosition - nominalState_.positionW;

		Eigen::Matrix<double, 6, 18> H = Eigen::Matrix<double, 6, 18>::Zero();
		H.block<3, 3>(0, thetaIndex) = Eigen::Matrix3d::Identity();
		H.block<3, 3>(3, positionIndex) = Eigen::Matrix3d::Identity();

		// Innovation covariance and chi^2 gate: rejects VO glitches before they
		// enter the state (the frame 4249 kind of spike).
		const Matrix6d S = H * covariance_ * H.transpose() + noise;
		const Eigen::LDLT<Matrix6d> Sldlt(S);
		result.residual = residual;
		result.mahalanobis2 = residual.dot(Sldlt.solve(residual));
		if (result.mahalanobis2 > chiSquareGate && !skipGateOnce_) return result;
		skipGateOnce_ = false;

		// K = P H^T S^-1
		const Eigen::Matrix<double, 18, 6> K = Sldlt.solve(H * covariance_).transpose();
		const Eigen::Matrix<double, 18, 1> dx = K * residual;

		// Joseph form keeps P symmetric positive definite.
		const CovarianceMatrix IKH = CovarianceMatrix::Identity() - K * H;
		covariance_ = IKH * covariance_ * IKH.transpose() + K * noise * K.transpose();
		covariance_ = 0.5 * (covariance_ + covariance_.transpose());

		// Inject the error into the nominal state (then δx = 0 again).
		nominalState_.quaternionWB = (nominalState_.quaternionWB * so3Exp(dx.segment<3>(thetaIndex))).normalized();
		nominalState_.positionW += dx.segment<3>(positionIndex);
		nominalState_.velocityW += dx.segment<3>(velocityIndex);
		nominalState_.biasGyro += dx.segment<3>(gyroBiasIndex);
		nominalState_.biasAcc += dx.segment<3>(accBiasIndex);
		nominalState_.gravityW += dx.segment<3>(gravityIndex);
		// (ESKF reset Jacobian I - [δθ/2]x is ~ I for small δθ and is skipped.)

		result.accepted = true;
		return result;
	}

	/*
		New reference frame W' = current body frame (R, p in the old frame W).
		  v' = R^T v,  g' = R^T g,  R' = I,  p' = 0
		Errors: with R_true = R Exp(δθ),  v'_true = Exp(-δθ) R^T (v + δv)
		  δv' = R^T δv + [v']x δθ,   δg' = R^T δg + [g']x δθ,   δθ' = δp' = 0
		P' = J P J^T with that J (biases unchanged).
	*/
	Eigen::Isometry3d Estimator::shiftReferenceFrame()
	{
		Eigen::Isometry3d motion = Eigen::Isometry3d::Identity();
		if (!stateInitialized_) return motion;
		const Eigen::Matrix3d R = nominalState_.quaternionWB.toRotationMatrix();
		motion.linear() = R;
		motion.translation() = nominalState_.positionW;

		const Eigen::Vector3d velocityNew = R.transpose() * nominalState_.velocityW;
		const Eigen::Vector3d gravityNew = R.transpose() * nominalState_.gravityW;

		CovarianceMatrix J = CovarianceMatrix::Zero();
		J.block<3, 3>(velocityIndex, velocityIndex) = R.transpose();
		J.block<3, 3>(velocityIndex, thetaIndex) = skew(velocityNew);
		J.block<3, 3>(gyroBiasIndex, gyroBiasIndex).setIdentity();
		J.block<3, 3>(accBiasIndex, accBiasIndex).setIdentity();
		J.block<3, 3>(gravityIndex, gravityIndex) = R.transpose();
		J.block<3, 3>(gravityIndex, thetaIndex) = skew(gravityNew);
		covariance_ = J * covariance_ * J.transpose();
		covariance_ = 0.5 * (covariance_ + covariance_.transpose());

		nominalState_.quaternionWB = Eigen::Quaterniond::Identity();
		nominalState_.positionW.setZero();
		nominalState_.velocityW = velocityNew;
		nominalState_.gravityW = gravityNew.normalized() * gravityMagnitude;  // keep |g| fixed
		return motion;
	}
}
