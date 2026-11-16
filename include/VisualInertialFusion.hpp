/* RigVis loosely coupled visual-inertial odometry.

   The stereo frontend runs exactly as before. Its frame-to-frame motion is fed
   as a measurement into the robocentric ESKF (Estimator), which propagates
   with every IMU sample in between. The fused motion of each frame is chained
   onto the previous fused pose. Output: one camera pose per frame, in the same
   world frame as the VO (first camera), evaluated like any other trajectory.

   Frames:  V = VO world (first left camera, x right, y down, z forward)
            C = rectified left camera,  B = IMU (OXTS, x forward, y left, z up)
            T_VB = T_VC * T_CB,  T_CB = calibration.camera_from_imu

   Initialization (needs ~2 s of tracking). The VO frame is not gravity
   aligned and the vehicle may already be moving, so gravity g and the initial
   velocity v0 are unknown. Two cases:

   Static start (VO barely moves, gyro ~ 0): the accelerometer measures only
   gravity:  g = -mean(R_VB f).  v0 = 0.

   Moving start: visual-inertial alignment by linear least squares (as in
   VINS-Mono). Integrating the IMU from the first window frame (rotation from
   the gyro, anchored at the VO rotation R_0) gives for every frame k
       p_k - p_0 = v0 t_k + 1/2 g t_k^2 + S_k,     S_k = double integral of R f
   which is linear in [v0, g]: 3 equations per frame, 6 unknowns, solved over
   all frames of the window. Then |g| is fixed to 9.81 and v0 re-solved.
   The fit residual tells whether the window was good (otherwise slide on).
*/
#pragma once

#include <deque>
#include <string>

#include <Eigen/Geometry>

#include "Estimator.hpp"
#include "types.hpp"

namespace RigVis
{
	struct VisualNoise
	{
		// 1 sigma of one frame-to-frame VO motion (tune from the RPE numbers).
		double rotationStdRad = 0.0035;  // ~0.2 deg
		double positionStdM = 0.03;
	};

	struct FusionFrameLog
	{
		bool initialized = false;
		bool updated = false;           // measurement applied
		bool rejected = false;          // failed the chi^2 gate
		double mahalanobis2 = 0.0;
		double residualRotationDeg = 0.0;
		double residualPositionM = 0.0;
		double speedMps = 0.0;
		Eigen::Vector3d biasGyroDegPerS = Eigen::Vector3d::Zero();
		Eigen::Vector3d biasAcc = Eigen::Vector3d::Zero();
	};

	class VisualInertialFusion
	{
		public:
			explicit VisualInertialFusion(const StereoCalibration& calibration,
										  double initializationWindowSeconds = 2.0);

			void addImu(const ImuSample& sample);

			// One call per stereo frame, after frontend.process().
			// visualPoseValid: the frontend tracked this frame (not lost / re-initialized)
			// and did not jump (loop correction): only then is the motion used.
			void addCameraFrame(double timestamp_s, const Eigen::Isometry3d& worldFromCamera, bool visualPoseValid);

			bool initialized() const { return estimator_.initialized(); }

			// Fused pose of the left camera in the VO world frame (falls back to the
			// VO pose before initialization).
			Eigen::Isometry3d cameraPose() const;

			const FusionFrameLog& lastLog() const { return log_; }
			const Estimator& estimator() const { return estimator_; }
			std::size_t updatesAccepted() const { return accepted_; }
			std::size_t updatesRejected() const { return rejected_; }

			void setVisualNoise(const VisualNoise& noise) { visualNoise_ = noise; }
			void setImuNoise(const ImuNoise& noise) { estimator_.setImuNoise(noise); }
			const ImuNoise& imuNoise() const { return estimator_.imuNoise(); }
			const VisualNoise& visualNoise() const { return visualNoise_; }

		private:
			struct WindowFrame
			{
				double timestamp_s;
				Eigen::Isometry3d worldFromBody;
			};
			struct WindowImu
			{
				double timestamp_s;
				Eigen::Vector3d acceleration;  // specific force, body frame
				Eigen::Vector3d angularVel;    // body frame
			};

			void tryInitialize();

			Estimator estimator_;
			Eigen::Isometry3d cameraFromImu_;
			Eigen::Isometry3d imuFromCamera_;
			double windowSeconds_;
			VisualNoise visualNoise_;

			std::deque<WindowFrame> windowFrames_;
			std::deque<WindowImu> windowImu_;

			// Body pose in the VO world, chained from the fused frame motions.
			Eigen::Isometry3d worldFromBody_ = Eigen::Isometry3d::Identity();

			bool havePreviousVisual_ = false;
			Eigen::Isometry3d previousWorldFromCamera_ = Eigen::Isometry3d::Identity();
			Eigen::Isometry3d lastVisualPose_ = Eigen::Isometry3d::Identity();

			FusionFrameLog log_;
			std::size_t accepted_ = 0;
			std::size_t rejected_ = 0;
	};

	std::string fusionCsvHeader();
	std::string fusionCsvRow(const FusionFrameLog& log);
}
