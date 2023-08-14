include(CMakeFindDependencyMacro)
find_package(OpenCV REQUIRED COMPONENTS core imgproc)
find_package(OpenVINO REQUIRED COMPONENTS Runtime)

include("${CMAKE_CURRENT_LIST_DIR}/model_apiTargets.cmake")
