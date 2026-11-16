#pragma once

#include "Estimator.hpp"

#include <Eigen/Core>

#include <vector>

namespace RigVis
{
    class TrajectoryViewer
    {
    public:
        void update(
            const FilterState& state,
            const CovarianceMatrix& covariance);

        void waitUntilClosed() const;

    private:
        std::vector<Eigen::Vector2d> pathXY_;
    };
}  // namespace RigVis