:# Prerequisites
:# 1. MSVC 2017 C++ Build Tools
:# 2. CMAKE 3.0 or up
:# 3. 64 bits of Windows
:# 4. Anaconda / MiniConda 64 bits
call "C:\Program Files (x86)\Microsoft Visual Studio\2017\Enterprise\VC\Auxiliary\Build\vcvars64.bat"
set path=C:\Program Files\Git\cmd;C:\Program Files\CMake\bin;%path%
:# Prerequisites for CUDA
:# 1. CUDA 8.0 or up
:# 2. NVTX( in CUDA as Visual Studio Integration. if fail to install, you can extract
:#     the CUDA installer exe and found the NVTX installer under the CUDAVisualStudioIntegration)
:# 3. MSVC 2015 with Update 2 or up for CUDA 8

:# Optional
:# 1. cuDNN 6.0 or up 
:# 2. Blas (MKL, OpenBlas)

:# If your main python version is < 3.5
set PYTHON_VERSION=3.6 
call conda create -q -n test python=%PYTHON_VERSION% numpy mkl cffi pyyaml
call activate test
conda install pybind11 -c conda-forge

git clone --recursive https://github.com/pytorch/pytorch
cd pytorch
xcopy /Y aten\src\ATen\common_with_cwrap.py tools\shared\cwrap_common.py

:# Open X86_64 Developer Command Prompt for MSVC 2017 and type in the following commands:
:# If you don't want to enable CUDA
set NO_CUDA=1

:# If multiple versions of CUDA are installed，it will build for the last one installed. To override
:#set CUDA_PATH=%CUDA_PATH_V8_0%
:#set PATH=%CUDA_PATH_V8_0%\bin;%PATH%
:# or
:set CUDA_PATH=%CUDA_PATH_V9_0%
:set PATH=%CUDA_PATH_V9_0%\bin;%PATH%

:# For CUDA 8 builds
:#set CMAKE_GENERATOR=Visual Studio 14 2015 Win64

:# For CUDA 9 / CPU builds
set CMAKE_GENERATOR=Visual Studio 15 2017 Win64

:# If you want to speed up CUDA builds, use Ninja
pip install ninja
set CMAKE_GENERATOR=Ninja
:# If you use Ninja with CUDA 8
:set PREBUILD_COMMAND=%VS140COMNTOOLS%\..\..\VC\vcvarsall.bat
:set PREBUILD_COMMAND_ARGS=x86_amd64

:# If you want to save time for rebuilding it next time, use clcache
pip install git+https://github.com/frerich/clcache.git
set USE_CLCACHE=1
set CC=clcache
set CXX=clcache

:# If you want to add BLAS(OpenBLAS, MKL)
set LIB=C:\Users\tlyons.BKP\AppData\Local\conda\conda\pkgs\mkl-2019.0-118\Library\bin;%LIB%
set LIB=C:\Users\tlyons.BKP\AppData\Local\conda\conda\pkgs\intel-openmp-2019.0-118\Library\bin;%LIB% 



:# If you have both VS2015 and VS2017
set DISTUTILS_USE_SDK=1
set USE_CUDA=1

:# Now you can start install PyTorch here
python setup.py bdist_wheel
