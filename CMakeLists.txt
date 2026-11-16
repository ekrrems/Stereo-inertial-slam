cmake_minimum_required(VERSION 3.30)

project(RigVis
    VERSION 0.1.0
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Default to an optimised build. Without this the build type is empty, which
# means no -O flags and Eigen/OpenCV-heavy code runs several times slower.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
	set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
message(STATUS "RigVis build type: ${CMAKE_BUILD_TYPE}")

find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS
	opencv_core
	opencv_imgcodecs
	opencv_highgui
	opencv_imgproc
	opencv_features
	opencv_calib
)
# Local bundle adjustment. macOS: brew install ceres-solver
find_package(Ceres REQUIRED)

add_library(rigvis
    src/DataLoader.cpp
	src/Estimator.cpp
	src/VisualInertialFusion.cpp
	src/TrajectoryViewer.cpp
	src/StereoFrontend.cpp
	src/MapViewer.cpp
	src/Evaluation.cpp
	src/LocalBundleAdjuster.cpp
	src/PoseGraph.cpp
	src/LoopClosing.cpp
	src/DenseMapper.cpp
	src/MapExport.cpp
)

target_include_directories(rigvis
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(rigvis
    PUBLIC
        Eigen3::Eigen
        ${OpenCV_LIBS}
        Ceres::ceres
)

target_compile_options(rigvis
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

add_executable(rigvis_dataset_runner
    src/main.cpp
)

target_link_libraries(rigvis_dataset_runner
    PRIVATE
        rigvis
)

# Default config location (the file itself is git-ignored; only the path is
# compiled into the binary, it never appears in the repository).
target_compile_definitions(rigvis_dataset_runner
    PRIVATE
        RIGVIS_DEFAULT_CONFIG="${CMAKE_CURRENT_SOURCE_DIR}/config/rigvis.yaml"
)

# Re-evaluate a saved trajectory without rerunning the dataset:
#   ./rigvis_evaluate trajectory_kitti.txt <..._sync folder> [first_frame] [plot.png]
add_executable(rigvis_evaluate
    tools/rigvis_evaluate.cpp
)

target_link_libraries(rigvis_evaluate
    PRIVATE
        rigvis
)
