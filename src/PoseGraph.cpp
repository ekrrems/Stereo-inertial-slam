#include "PoseGraph.hpp"

#include <array>
#include <chrono>

#include <ceres/ceres.h>

namespace RigVis
{
	namespace
	{
		// One relative-pose edge. Parameters: q_i (4, x y z w), t_i (3), q_j (4), t_j (3),
		// all describing T_wc (world from camera).
		struct RelativePoseError
		{
			RelativePoseError(const Eigen::Isometry3d& measured, double translationSigma, double rotationSigma)
				: measuredTranslation_(measured.translation()),
				  measuredRotation_(Eigen::Quaterniond(measured.linear()).normalized()),
				  inverseTranslationSigma_(1.0 / translationSigma),
				  inverseRotationSigma_(1.0 / rotationSigma) {}

			template <typename T>
			bool operator()(const T* const qi, const T* const ti, const T* const qj, const T* const tj,
							T* residual) const
			{
				const Eigen::Map<const Eigen::Quaternion<T>> rotationI(qi), rotationJ(qj);
				const Eigen::Map<const Eigen::Matrix<T, 3, 1>> translationI(ti), translationJ(tj);

				// Estimated relative pose T_ij = T_i^-1 * T_j
				const Eigen::Quaternion<T> rotationIInverse = rotationI.conjugate();
				const Eigen::Quaternion<T> relativeRotation = rotationIInverse * rotationJ;
				const Eigen::Matrix<T, 3, 1> relativeTranslation = rotationIInverse * (translationJ - translationI);

				// Difference to the measurement.
				const Eigen::Quaternion<T> rotationError =
					measuredRotation_.template cast<T>() * relativeRotation.conjugate();

				Eigen::Map<Eigen::Matrix<T, 6, 1>> r(residual);
				r.template head<3>() = (relativeTranslation - measuredTranslation_.template cast<T>()) *
									   T(inverseTranslationSigma_);
				r.template tail<3>() = T(2.0) * rotationError.vec() * T(inverseRotationSigma_);
				return true;
			}

			static ceres::CostFunction* create(const Eigen::Isometry3d& measured, double translationSigma,
											   double rotationSigma)
			{
				return new ceres::AutoDiffCostFunction<RelativePoseError, 6, 4, 3, 4, 3>(
					new RelativePoseError(measured, translationSigma, rotationSigma));
			}

			Eigen::Vector3d measuredTranslation_;
			Eigen::Quaterniond measuredRotation_;
			double inverseTranslationSigma_;
			double inverseRotationSigma_;
		};
	}

	PoseGraphSummary optimizePoseGraph(std::vector<Eigen::Isometry3d>& worldFromCamera,
									   const std::vector<PoseGraphEdge>& edges,
									   const std::vector<bool>& fixed,
									   const int maxIterations)
	{
		const auto startTime = std::chrono::steady_clock::now();
		PoseGraphSummary summary;
		const std::size_t n = worldFromCamera.size();
		if (n < 2 || edges.empty()) return summary;

		std::vector<std::array<double, 4>> quaternions(n);
		std::vector<std::array<double, 3>> translations(n);
		for (std::size_t i = 0; i < n; ++i) {
			const Eigen::Quaterniond q(worldFromCamera[i].linear());
			quaternions[i] = {q.x(), q.y(), q.z(), q.w()};
			translations[i] = {worldFromCamera[i].translation().x(), worldFromCamera[i].translation().y(),
							   worldFromCamera[i].translation().z()};
		}

		ceres::Problem problem;
		for (const PoseGraphEdge& edge : edges) {
			if (edge.from >= n || edge.to >= n || edge.from == edge.to) continue;
			problem.AddResidualBlock(
				RelativePoseError::create(edge.measuredFromTo, edge.translationSigma, edge.rotationSigma),
				nullptr,  // no robust loss: loops are verified before they get here
				quaternions[edge.from].data(), translations[edge.from].data(),
				quaternions[edge.to].data(), translations[edge.to].data());
			++summary.edges;
		}

		for (std::size_t i = 0; i < n; ++i) {
			if (!problem.HasParameterBlock(quaternions[i].data())) continue;
			problem.SetManifold(quaternions[i].data(), new ceres::EigenQuaternionManifold);
			if (i < fixed.size() && fixed[i]) {
				problem.SetParameterBlockConstant(quaternions[i].data());
				problem.SetParameterBlockConstant(translations[i].data());
			}
			++summary.nodes;
		}

		ceres::Solver::Options options;
		options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;  // chain/graph structure is very sparse
		options.max_num_iterations = maxIterations;
		options.num_threads = 4;
		options.logging_type = ceres::SILENT;

		ceres::Solver::Summary ceresSummary;
		ceres::Solve(options, &problem, &ceresSummary);

		for (std::size_t i = 0; i < n; ++i) {
			worldFromCamera[i].linear() = Eigen::Quaterniond(quaternions[i][3], quaternions[i][0], quaternions[i][1],
															 quaternions[i][2]).normalized().toRotationMatrix();
			worldFromCamera[i].translation() = Eigen::Vector3d(translations[i][0], translations[i][1], translations[i][2]);
		}

		summary.ran = true;
		summary.iterations = static_cast<int>(ceresSummary.iterations.size());
		summary.initialCost = ceresSummary.initial_cost;
		summary.finalCost = ceresSummary.final_cost;
		summary.milliseconds =
			std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - startTime).count();
		return summary;
	}
}
