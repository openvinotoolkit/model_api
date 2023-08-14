include(CMakeFindDependencyMacro)
find_dependency(OpenCV REQUIRED COMPONENTS core imgproc)
find_dependency(OpenVINO REQUIRED COMPONENTS Runtime)

include("${CMAKE_CURRENT_LIST_DIR}/model_apiTargets.cmake")
