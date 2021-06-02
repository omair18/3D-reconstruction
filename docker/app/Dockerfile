FROM nvidia/cuda:11.3.0-devel-ubuntu20.04

ENV TZ=Europe/Minsk
ENV NVIDIA_DRIVER_CAPABILITIES=all

RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime \
    && echo $TZ > /etc/timezone \
    && apt-get -y update \ 
    && apt-get install -y \
        apt-utils \
        software-properties-common \
    && add-apt-repository "deb http://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /" \
    && apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub \
    && apt -y update \
    && apt install -y \
        autoconf \
        build-essential \
        zlib1g-dev \
        git \
        gcc-9 \
        g++-9 \
        clang-9 \
        nasm \
        yasm \
        zip \
        unzip \
        pkg-config \
        wget \
        libcurl4-gnutls-dev \
        libcutensor1 \
        libnvgraph10 \
        libcutensor-dev \
        libpng-dev \
        libjpeg-turbo8-dev \
        libmetis-dev \
        libtiff-dev \
        libxxf86vm1 \
        libatlas-base-dev \
        libsuitesparse-dev \
        libxxf86vm-dev \
        libxi-dev \
        libxrandr-dev \
        graphviz \
        liblz4-1 \
        liblz4-dev \
        libopenexr-dev \
        libssl-dev \
        libsasl2-dev \
        bzip2 \
        libzstd-dev \
        libpthread-stubs0-dev \
        libpthread-workqueue-dev \
        libpthread-workqueue0 \
        libgmp-dev \
        libgmp3-dev \
        libmpfrc++-dev \
        libsuperlu-dev \
        libfftw3-dev \
        libzip-dev \
        libbz2-dev \
        libbzrtp-dev \  
        libopenmpt-dev \
        libvpx-dev \
        libx265-dev \
        libx264-dev \
        libwebp-dev \
        libxvidcore-dev \
        libdc1394-22-dev \
        libtheora-dev \
        pkg-config \
        libdrm-dev \
        libopenjp2-7-dev \
    && ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1 \
    && rm /usr/bin/gcc \
    && rm /usr/bin/g++ \
    && ln -s /usr/bin/gcc-9 /usr/bin/gcc \
    && ln -s /usr/bin/g++-9 /usr/bin/g++ \
    && ln -s /usr/bin/clang++-9 /usr/bin/clang++ \
    && ln -s /usr/bin/clang-9 /usr/bin/clang \
    && apt purge -y \
        libeigen3-dev \
        libgoogle-glog-dev \
        libgflags-dev \
        cmake \
    && rm -rf /var/lib/apt/lists/*
    
RUN cd /opt \
    && wget https://cmake.org/files/v3.19/cmake-3.19.7.tar.gz \
    && tar -xzvf cmake-3.19.7.tar.gz \
    && rm cmake-3.19.7.tar.gz \
    && cd cmake-3.19.7 \
    && ./bootstrap \
    && make install -j`nproc` \
    && rm -rf /opt/cmake-3.19.7
    
RUN cd /opt \
    && mkdir NvCodecHeaders \
    && cd NvCodecHeaders \
    && git clone --branch=n11.0.10.1 https://github.com/FFmpeg/nv-codec-headers.git \
    && cd nv-codec-headers \
    && make -j`nproc` \
    && make install \
    && rm -rf /opt/NvCodecHeaders
    
RUN cd /opt \
    && mkdir FFmpeg \
    && cd FFmpeg \
    && git clone --branch=n4.3.2 https://github.com/FFmpeg/FFmpeg.git \
    && cd FFmpeg \
    && ./configure \
        --prefix=/usr \
        --libdir=/usr/lib/x86_64-linux-gnu \
        --incdir=/usr/include/x86_64-linux-gnu \
        --arch=amd64 \
        --enable-gpl \
        --enable-nonfree \
        --enable-libopenjpeg \
        --enable-libopenmpt \
        --enable-libvpx \
        --enable-libwebp \
        --enable-libxml2 \
        --enable-libxvid \
        --enable-pthreads \
        --enable-libdc1394 \
        --enable-libdrm \
        --enable-static \
        --enable-shared \
        --enable-nvenc \
        --enable-postproc \
        --enable-libx264 \
        --enable-libx265 \
        --enable-libtheora \
        --enable-cuda-nvcc \
        --enable-libnpp \
        --enable-pic \
        --extra-libs='-lpthread -lm' \
        --extra-cflags="-I/usr/local/cuda/include" \
        --extra-ldflags="-L/usr/local/cuda/lib64" \
        --nvccflags="-gencode arch=compute_86,code=[sm_86,compute_86] -O3" \
    && make -j`nproc` \
    && make install \
    && rm -r /opt/FFmpeg
    
RUN cd /opt \
    && git clone https://github.com/edenhill/librdkafka.git --branch=v1.7.0 \
    && cd librdkafka \
    && git submodule init \
    && git submodule update \
    && mkdir build \
    && cd build \
    && cmake \
        -DCMAKE_CXX_STANDARD=17 \
        -DCMAKE_CXX_STANDARD_REQUIRED=ON \
        -DCMAKE_BUILD_TYPE=RELEASE \
        -DBUILD_SHARED_LIBS=ON \
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
        -DCMAKE_CXX_FLAGS="-Ofast -march=native -mtune=native -std=gnu++17 -fPIC -fopenmp -fopenmp-simd -DNDEBUG" \
        -DCMAKE_C_FLAGS="-Ofast -march=native -mtune=native -fPIC -fopenmp -fopenmp-simd -DNDEBUG" \
        -DCMAKE_INSTALL_PREFIX=/var/3DReconstructionEnv/librdkafka \
        -DRDKAFKA_BUILD_EXAMPLES=OFF \
        -DRDKAFKA_BUILD_TESTS=OFF \
        .. \
    && make -j`nproc` \
    && make install \
    && rm -r /opt/librdkafka

RUN cd /opt \
    && mkdir boost_lib \
    && mkdir -p /var/3DReconstructionEnv/boost \
    && cd boost_lib \
    && wget https://boostorg.jfrog.io/artifactory/main/release/1.76.0/source/boost_1_76_0.tar.gz \
    && tar -xzvf boost_1_76_0.tar.gz \
    && rm boost_1_76_0.tar.gz \
    && cd boost_1_76_0 \
    && ./bootstrap.sh --prefix=/var/3DReconstructionEnv/boost \
    && ./b2 install cxxflags="-fPIC -Ofast -march=native -mtune=native -std=gnu++17 -fopenmp -fopenmp-simd -DNDEBUG" \
    && rm -r /opt/boost_lib
   
RUN cd /opt \
    && git clone https://gitlab.com/libeigen/eigen.git --branch=3.3 \
    && cd eigen \
    && git submodule init \
    && git submodule update \
    && sed -i -e 's/-std=c++11/-std=gnu++17/' CMakeLists.txt \
    && sed -i -e '/CMAKE_CXX_STANDARD/s/11/17/' CMakeLists.txt \
    && sed -i -e '/CMAKE_CXX_EXTENSIONS/s/OFF/ON/' CMakeLists.txt \
    && sed -i -e '/EIGEN_CUDA_COMPUTE_ARCH/s/30/86/' CMakeLists.txt \
    && mkdir build \
    && cd build \
    && cmake \
        -DCMAKE_BUILD_TYPE=RELEASE \
        -DCMAKE_CXX_STANDARD=17 \
        -DBOOST_INCLUDEDIR=/var/3DReconstructionEnv/boost/include \
        -DBOOST_LIBRARYDIR=/var/3DReconstructionEnv/boost/lib \
        -DCMAKE_CXX_STANDARD_REQUIRED=ON \
        -DCMAKE_CXX_FLAGS="-Ofast -march=native -mtune=native -fPIC -fopenmp -fopenmp-simd -DNDEBUG" \
        -DCMAKE_C_FLAGS="-Ofast -march=native -mtune=native -fPIC -fopenmp -fopenmp-simd -DNDEBUG" \
        -DEIGEN_CUDA_COMPUTE_ARCH=86 \
        -DCUDA_NVCC_FLAGS="-gencode arch=compute_86,code=[sm_86,compute_86]" \
        -DBUILD_TESTING=OFF \
        -DBUILD_SHARED_LIBS=ON \
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
        -DEIGEN_TEST_CXX11=ON \
        -DEIGEN_BUILD_BTL=OFF \
        -DCMAKE_INSTALL_PREFIX=/var/3DReconstructionEnv/eigen \
        .. \
    && make -j`nproc` \
    && make install \
    && rm -rf /opt/eigen
        
RUN cd /opt \
    && mkdir opencv_lib \
    && cd opencv_lib \
    && git clone https://github.com/opencv/opencv.git --branch=4.5.1 \
    && git clone https://github.com/opencv/opencv_contrib.git --branch=4.5.1 \
    && cd opencv \
    && mkdir build \
    && cd build \
    && cmake \
        -DBUILD_EXAMPLES=OFF \
        -DBUILD_PERF_TESTS=OFF \
        -DBUILD_SHARED_LIBS=ON \
        -DCMAKE_CXX_STANDARD=17 \
        -DENABLE_CXX11=ON \
        -DCMAKE_CXX_STANDARD_REQUIRED=ON \
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
        -DCMAKE_CXX_FLAGS="-Ofast -march=native -mtune=native -fPIC -DNDEBUG" \
        -DCMAKE_CXX_FLAGS_RELEASE="-Ofast -march=native -mtune=native -fPIC -DNDEBUG" \
        -DCMAKE_C_FLAGS="-Ofast -march=native -mtune=native -fPIC -DNDEBUG" \
        -DCMAKE_C_FLAGS_RELEASE="-Ofast -march=native -mtune=native -fPIC -DNDEBUG" \
        -DBUILD_TESTS=OFF \
        -DCMAKE_LIBRARY_PATH=/usr/local/cuda/lib64/stubs \
        -DCMAKE_INSTALL_PREFIX=/var/3DReconstructionEnv/opencv \
        -DCMAKE_BUILD_TYPE=RELEASE \
        -DOPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \
        -DWITH_CUDA=ON \
        -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda \
        -DENABLE_FAST_MATH=ON \
        -DCUDA_FAST_MATH=ON \
        -DWITH_CUBLAS=ON \
        -DWITH_CUFFT=ON \
        -DWITH_EIGEN=ON \
        -DWITH_VULKAN=OFF \
        -DBUILD_opencv_python3=OFF \
        -DBUILD_opencv_python_tests=OFF \
        -DBUILD_opencv_python_bindings_generator=OFF \
        -DBUILD_opencv_optflow=OFF \
        -DBUILD_opencv_aruco=OFF \
        -DBUILD_opencv_sfm=OFF \
        -DBUILD_opencv_dnn=OFF \
        -DBUILD_opencv_apps=OFF \
        -DWITH_OPENMP=ON \
        -DWITH_IPP=ON \
        -DWITH_OPENEXR=ON \
        -DWITH_OPENGL=OFF \
        -DWITH_QT=OFF \
        -DWITH_TBB=ON \
        -DWITH_V4L=ON \
        -DWITH_FFMPEG=ON \
        -DWITH_QUIRC=OFF \
        -DBUILD_opencv_ml=OFF \
        -DBUILD_opencv_bioinspired=OFF \
        -DBUILD_opencv_rapid=OFF \
        -DBUILD_IPP_IW=ON \
        -DOPENCV_ENABLE_NONFREE=ON \
        -DCUDA_ARCH_BIN=8.6 \
        -DCUDA_ARCH_PTX=8.6 \
        -DENABLE_PRECOMPILED_HEADERS=OFF \
        -DEigen3_DIR=/var/3DReconstructionEnv/eigen/share/eigen3/cmake \
        .. \
    && make -j`nproc` \
    && make install \
    && rm -rf /opt/opencv_lib
    
RUN cd /opt \
    && git clone https://github.com/openMVG/openMVG.git --branch=v1.6 \
    && cd openMVG \
    && git submodule init \
    && git submodule update \
    && mkdir build \
    && cd build \
    && cmake \
        -DCMAKE_BUILD_TYPE=RELEASE \
        -DCMAKE_CXX_FLAGS="-Ofast -fopenmp -march=native -mtune=native -std=gnu++17 -fPIC -fopenmp-simd -DNDEBUG" \
        -DCMAKE_C_FLAGS="-Ofast -fopenmp -march=native -mtune=native -fPIC -fopenmp-simd -DNDEBUG" \
        -DOpenMVG_BUILD_DOC=OFF \
        -DOpenMVG_BUILD_EXAMPLES=OFF \
        -DOpenMVG_BUILD_GUI_SOFTWARES=OFF \
        -DOpenMVG_BUILD_SOFTWARES=OFF \
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
        -DOpenMVG_BUILD_SHARED=ON \
        -DEigen3_DIR=/var/3DReconstructionEnv/eigen/share/eigen3/cmake \
        -DCMAKE_INSTALL_PREFIX=/var/3DReconstructionEnv/openMVG \
        -DEIGEN_INCLUDE_DIR_HINTS=/var/3DReconstructionEnv/eigen/include \
        ../src/ \
    && make -j`nproc` \
    && make install \
    && rm -rf /opt/openMVG
    
RUN cd /opt \
    && git clone https://github.com/CGAL/cgal.git --branch=v5.2 \
    && cd cgal \
    && git submodule init \
    && git submodule update \
    && mkdir build \
    && cd build \
    && cmake \
        -DBoost_INCLUDE_DIR=/var/3DReconstructionEnv/boost/include/ \
        -DCMAKE_BUILD_TYPE=Release \
        -DCGAL_HEADER_ONLY=OFF \
        -DCMAKE_CXX_STANDARD=17 \
        -DCMAKE_CXX_STANDARD_REQUIRED=ON \
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
        -DOpenMVG_BUILD_SHARED=ON \
        -DCMAKE_CXX_FLAGS="-Ofast -march=native -mtune=native -std=gnu++17 -fPIC -fopenmp -fopenmp-simd -DNDEBUG" \
        -DCMAKE_C_FLAGS="-Ofast -march=native -mtune=native -fPIC -fopenmp -fopenmp-simd -DNDEBUG" \
        -DCMAKE_INSTALL_PREFIX=/var/3DReconstructionEnv/CGAL \
        -DEigen3_DIR=/var/3DReconstructionEnv/eigen/share/eigen3/cmake \
        .. \
    && make -j`nproc` \
    && make install \
    && rm -rf /opt/cgal
    
RUN cd /var/3DReconstructionEnv \
    && git clone https://github.com/cdcseacave/VCG.git --branch=v1.0.1 vcglib
    
RUN cd /opt \
    && git clone https://github.com/cdcseacave/openMVS.git --branch=v1.1.1 \
    && cd openMVS \
    && sed -i -e '/FIND_PACKAGE(OpenCV/s/)/\ PATHS \/var\/3DReconstructionEnv\/opencv NO_DEFAULT_PATH)/' CMakeLists.txt \
    && sed -i -e '/FIND_PACKAGE(Eige/s/n/n3/' CMakeLists.txt \
    && sed -i -e '/FIND_PACKAGE(Eigen3/s/)/\ PATHS \/var\/3DReconstructionEnv\/eigen NO_DEFAULT_PATH)/' CMakeLists.txt \
    && sed -i -e 's/EIGEN_FOUND/EIGEN3_FOUND/' CMakeLists.txt \
    && sed -i -e 's/EIGEN_INCLUDE_DIRS/EIGEN3_INCLUDE_DIRS/' CMakeLists.txt \
    && mkdir build_mvs \
    && cd build_mvs \
    && cmake \
        -DCMAKE_BUILD_TYPE=Release \
        -DBOOST_INCLUDEDIR=/var/3DReconstructionEnv/boost/include \
        -DBOOST_LIBRARYDIR=/var/3DReconstructionEnv/boost/lib \
        -DVCG_ROOT=/var/3DReconstructionEnv/vcglib \
        -DOpenMVS_USE_BREAKPAD=OFF \
        -DBUILD_SHARED_LIBS=OFF \
        -DCMAKE_CXX_STANDARD=17 \
        -DCMAKE_CXX_STANDARD_REQUIRED=ON \
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
        -DOpenMVS_USE_CUDA=ON \
        -DCMAKE_CXX_FLAGS="-Ofast -march=native -mtune=native -std=gnu++17 -fPIC -fopenmp -fopenmp-simd -DNDEBUG" \
        -DCMAKE_C_FLAGS="-Ofast -march=native -mtune=native -fPIC -fopenmp -fopenmp-simd -DNDEBUG" \
        -DCMAKE_INSTALL_PREFIX=/var/3DReconstructionEnv/openMVS \
        -DCGAL_ROOT=/var/3DReconstructionEnv/CGAL/lib/cmake/CGAL \
        -DCMAKE_LIBRARY_PATH=/usr/local/cuda/lib64/stubs \
        .. \
    && make -j`nproc` \
    && make install \
    && rm -rf /opt/openMVS

#RUN cd /opt \
#    && git clone https://github.com/aws/aws-sdk-cpp.git --branch=1.8.161 \
#    && cd aws-sdk-cpp \
#    && git submodule init \
#    && git submodule update \
#    && mkdir build \
#    && cd build \
#    && cmake \
#        -DCMAKE_CXX_STANDARD=17 \
#        -DCMAKE_CXX_STANDARD_REQUIRED=ON \
#        -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
#        -DCMAKE_CXX_FLAGS="-Ofast -march=native -mtune=native -std=gnu++17 -fPIC -fopenmp -fopenmp-simd -DNDEBUG" \
#        -DCMAKE_CXX_FLAGS_RELEASE="-Ofast -march=native -mtune=native -std=gnu++17 -fPIC -fopenmp -fopenmp-simd -DNDEBUG" \
#        -DCMAKE_C_FLAGS="-Ofast -march=native -mtune=native -fPIC -fopenmp -fopenmp-simd -DNDEBUG" \
#        -DCMAKE_C_FLAGS_RELEASE="-Ofast -march=native -mtune=native -fPIC -fopenmp -fopenmp-simd -DNDEBUG" \
#        -DCMAKE_BUILD_TYPE=RELEASE \
#        -DBUILD_SHARED_LIBS=ON \
#        -DCMAKE_INSTALL_PREFIX=/var/3DReconstructionEnv/aws \
#        -DENABLE_UNITY_BUILD=OFF \
#        -DENABLE_TESTING=OFF \
#        -DAUTORUN_UNIT_TESTS=OFF \
#        -DANDROID_BUILD_CURL=OFF \
#        -DANDROID_BUILD_OPENSSL=OFF \
#        -DANDROID_BUILD_ZLIB=OFF \
#        .. \
#    && make -j`nproc` \
#    && make install \
#    && rm -rf /opt/aws-sdk-cpp

COPY 3D-reconstruction /opt/3D-reconstruction

RUN cd /opt/3D-reconstruction \
    && mkdir build \
    && cd build \
    && cmake \
        -DCMAKE_CXX_STANDARD=17 \
        -DCMAKE_CXX_STANDARD_REQUIRED=ON \
        -DCMAKE_BUILD_TYPE=RELEASE \
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
        -DBUILD_SHARED_LIBS=ON \
        .. \
    && make -j`nproc`

WORKDIR /opt/3D-reconstruction/build/bin
