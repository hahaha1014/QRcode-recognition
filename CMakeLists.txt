cmake_minimum_required(VERSION 3.10) 

set(CMAKE_TOOLCHAIN_FILE E:/vcpkg/scripts/buildsystems/vcpkg.cmake
  CACHE STRING "Vcpkg toolchain file")

 set(ZBAR_INCLUDE_PATH "E:/ZBar/include")
 set(ZBAR_LIB_DIR "E:/ZBar/lib")
 set(ZBAR_DIR "E:/ZBar/bin")

# set the project name 
project(QRCODE)

find_package(OpenCV REQUIRED)
# Additional Include Directories
include_directories( ${OpenCV_INCLUDE_DIRS}
	${ZBAR_INCLUDE_PATH})

# Additional Library Directories
link_directories(${ZBAR_LIB_DIR}
	${ZBAR_DIR})

# add the executable 
add_executable(qrcode main.cpp)

# Additional DependenciesV
target_link_libraries(qrcode ${OpenCV_LIBS} libzbar-0 libzbar64-0)
