#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace RigVis
{
    inline Eigen::Matrix3d skew(const Eigen::Vector3d& vector)
    {
        Eigen::Matrix3d matrix;

        matrix <<
             0.0,       -vector.z(),  vector.y(),
             vector.z(), 0.0,         -vector.x(),
            -vector.y(), vector.x(),   0.0;

        return matrix;
    }

    // Exp: rotation vector (axis * angle) -> quaternion.
    inline Eigen::Quaterniond so3Exp(const Eigen::Vector3d& rotationVector)
    {
        const double angle = rotationVector.norm();
        if (angle < 1e-12) {
            // First order: q = [1, v/2], normalized.
            return Eigen::Quaterniond(1.0, 0.5 * rotationVector.x(), 0.5 * rotationVector.y(),
                                      0.5 * rotationVector.z()).normalized();
        }
        return Eigen::Quaterniond(Eigen::AngleAxisd(angle, rotationVector / angle));
    }

    // Log: quaternion -> rotation vector (axis * angle), angle in [0, pi].
    inline Eigen::Vector3d so3Log(const Eigen::Quaterniond& quaternion)
    {
        const Eigen::AngleAxisd angleAxis(quaternion.normalized());
        double angle = angleAxis.angle();
        Eigen::Vector3d axis = angleAxis.axis();
        if (angle > EIGEN_PI) {  // take the short way round
            angle = 2.0 * EIGEN_PI - angle;
            axis = -axis;
        }
        return axis * angle;
    }
}  // namespace RigVis
