cmake_minimum_required(VERSION 3.12)
project(ApproxOrtho)

# Установка C++ стандарта
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Добавляем поддержку pybind11
include(FetchContent)

cmake_policy(SET CMP0148 NEW)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
# Загружаем и устанавливаем pybind11
FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG v2.10.0  # Или последнюю версию
)
FetchContent_MakeAvailable(pybind11)

# Загружаем и устанавливаем Eigen
FetchContent_Declare(
  eigen
  GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
  GIT_TAG 3.4.0  # Или последнюю стабильную версию
)
FetchContent_MakeAvailable(eigen)


# Добавляем исходный файл
pybind11_add_module(approx_orto approx.cpp approx_orto.cpp)
add_executable(approx_orto_main main.cpp approx_orto.cpp )
# Указываем, что используем pybind11 и Eigen
target_link_libraries(approx_orto PRIVATE pybind11::module Eigen3::Eigen)
target_link_libraries(approx_orto_main PRIVATE pybind11::module Eigen3::Eigen)
target_link_libraries(approx_orto PRIVATE Python3::Python)


# Устанавливаем флаги компилятора для создания модуля Python
#set_target_properties(approx_orto PROPERTIES PREFIX "" SUFFIX ".so")

# Указываем, куда будут установлены зависимости
install(TARGETS approx_orto DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/python_module)
