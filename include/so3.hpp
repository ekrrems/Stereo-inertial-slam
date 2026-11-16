#pragma once

#include <Eigen/Core>

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
}  // namespace RigVis