ARG BASE_IMAGE="pytorch/pytorch:latest"
FROM $BASE_IMAGE as librealsense-builder

################################
#   Librealsense Builder Stage  #
#################################

ARG LIBRS_VERSION="2.56.3"
RUN test -n "$LIBRS_VERSION"

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -qq -y --no-install-recommends \
    build-essential cmake git libssl-dev libusb-1.0-0-dev pkg-config \
    libgtk-3-dev libglfw3-dev libgl1-mesa-dev libglu1-mesa-dev \
    curl python3 python3-dev ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /usr/src
RUN curl https://codeload.github.com/IntelRealSense/librealsense/tar.gz/refs/tags/v$LIBRS_VERSION -o librealsense.tar.gz
RUN tar -zxf librealsense.tar.gz && rm librealsense.tar.gz
RUN ln -s /usr/src/librealsense-$LIBRS_VERSION /usr/src/librealsense

ENV CMAKE_POLICY_VERSION_MINIMUM=3.5
RUN cd /usr/src/librealsense && mkdir build && cd build \
    && cmake -DCMAKE_C_FLAGS_RELEASE="${CMAKE_C_FLAGS_RELEASE} -s" \
    -DCMAKE_CXX_FLAGS_RELEASE="${CMAKE_CXX_FLAGS_RELEASE} -s" \
    -DCMAKE_INSTALL_PREFIX=/opt/librealsense \
    -DBUILD_GRAPHICAL_EXAMPLES=OFF -DBUILD_PYTHON_BINDINGS:bool=true \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
    -DPYTHON_EXECUTABLE:string=/opt/conda/bin/python3 .. \
    && make -j4 all && make install

######################################
#   librealsense Base Image Stage    #
######################################
FROM ${BASE_IMAGE} as librealsense

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx libx11-xcb1 libusb-1.0-0 udev curl ca-certificates \
    && apt-get clean all && rm -r /var/lib/apt/lists/*

RUN pip install --upgrade pip setuptools

RUN pip install torch torchvision einops kornia huggingface_hub \
    opencv-python-headless timm scipy scikit-image numpy==1.25.2 \
    astropy matplotlib pandas scikit-learn paho-mqtt pyrealsense2

COPY --from=librealsense-builder /opt/librealsense /usr/local/
COPY --from=librealsense-builder /usr/src/librealsense/config/99-realsense-libusb.rules /etc/udev/rules.d/
COPY --from=librealsense-builder /usr/src/librealsense/config/99-realsense-d4xx-mipi-dfu.rules /etc/udev/rules.d/
ENV PYTHONPATH=$PYTHONPATH:/usr/local/lib

WORKDIR /home/workspace
