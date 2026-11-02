# =============================================================================
# RiemannEngine C++ — Debian Package Builder
# =============================================================================
#
# Builds libriemann-cpp-dev_<VERSION>_cu<PROFILE>_ubuntu<UBUNTU>_amd64.deb
# inside the selected Ubuntu/CUDA environment, plus a matching SDK tarball
# RiemannEngine-<VERSION>-cu<PROFILE>-ubuntu<UBUNTU>-x64.tar.gz, and drops both
# at /output/ so the host script can copy them out.
#
# The package installs to /usr/local/RiemannEngine/<VERSION>_cu<PROFILE>/ and
# is named libriemann-cpp-dev-<VERSION>-cu<PROFILE>, so every version/CUDA
# combination can be installed side by side.
#
# Build args
# ----------
#   VERSION     Semver string embedded in the package (default: 1.1.0)
#   JOBS        Parallel ninja jobs (default: 4)
#   CUDA_VERSION  124, 132, or 134 (default: 134)
#   CUDA_ARCHITECTURES  CMake GPU architectures (default: 86)
#
# =============================================================================

ARG VERSION=1.1.0
ARG JOBS=4
ARG CUDA_VERSION=134

FROM nvidia/cuda:12.4.0-devel-ubuntu22.04 AS cuda-124
FROM nvidia/cuda:13.2.1-cudnn-devel-ubuntu24.04 AS cuda-132
FROM nvidia/cuda:13.4.3-cudnn-devel-ubuntu24.04 AS cuda-134

# ---------------------------------------------------------------------------
# Stage 1 – builder
# ---------------------------------------------------------------------------
FROM cuda-${CUDA_VERSION} AS builder

# Re-declare ARGs inside each stage so they are accessible in RUN commands.
# Promote VERSION to ENV so it can be referenced in later ENV and RUN lines.
ARG VERSION
ARG JOBS
ARG CUDA_VERSION
ARG CUDA_ARCHITECTURES=86
ENV VERSION=${VERSION}
ENV CUDA_VERSION=${CUDA_VERSION}

ENV DEBIAN_FRONTEND=noninteractive

# Install build tools
RUN apt-get update && apt-get install -y \
    build-essential \
    ninja-build \
    wget \
    unzip \
    libomp-dev \
    dpkg-dev \
    && rm -rf /var/lib/apt/lists/*

# CUDA 13 needs modern CMake; Torch 2.5.1 still needs legacy NvToolsExt targets.
ARG CMAKE_VERSION
COPY docker/cuda-profile.sh /opt/cuda-profile.sh
RUN bash -c 'source /opt/cuda-profile.sh && \
    wget -q "https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-linux-x86_64.tar.gz" \
         -O /tmp/cmake.tar.gz && \
    tar -xzf /tmp/cmake.tar.gz -C /opt && \
    ln -sf /opt/cmake-${CMAKE_VERSION}-linux-x86_64/bin/* /usr/local/bin/ && \
    rm /tmp/cmake.tar.gz && \
    cmake --version'

# CUDA environment
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:/opt/libtorch/lib:${LD_LIBRARY_PATH}

# The Linux CUDA 13 archives use the CXX11 ABI without the old filename prefix.
RUN bash -c 'source /opt/cuda-profile.sh && wget -q "${LIBTORCH_URL}" -O /opt/libtorch.zip' && \
    cd /opt && \
    unzip -q libtorch.zip && \
    rm libtorch.zip

# Copy project source
WORKDIR /src
COPY . .

# Configure & build
RUN bash -c 'source /opt/cuda-profile.sh && \
    cmake -B build -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_PREFIX_PATH=/opt/libtorch \
      -DCMAKE_INSTALL_PREFIX=/usr/local \
      -DCMAKE_INSTALL_LIBDIR=lib \
      -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES}" \
      -DRIEMANN_ENABLE_GAUSSIAN=ON \
      -DRIEMANN_BUILD_TESTS=OFF \
      -DRIEMANN_BUILD_ADAPTIVE_TESTS=OFF \
      -DRIEMANN_BUILD_GAUSSIAN_BENCHMARKS=OFF \
    && cmake --build build --parallel ${JOBS}'

# ---------------------------------------------------------------------------
# Stage 2 – package
# ---------------------------------------------------------------------------
# Stay in the same image; run cmake --install into the deb staging tree, then
# produce the .deb with dpkg-deb.

# Directories
ENV PKG_NAME=libriemann-cpp-dev-${VERSION}-cu${CUDA_VERSION}
ENV INSTALL_SUBDIR=${VERSION}_cu${CUDA_VERSION}
ENV STAGE=/deb-stage
ENV DEB_ROOT=${STAGE}/${PKG_NAME}_${VERSION}_amd64
ENV PKG_PREFIX=${DEB_ROOT}/usr/local/RiemannEngine/${INSTALL_SUBDIR}

# cmake --install into the staging tree
RUN cmake --install build --prefix ${PKG_PREFIX}

# Flatten the static archive so all symbols are in one thin ar member
# (prevents "archive has no index" errors for some linkers)
RUN ld -r --whole-archive ${PKG_PREFIX}/lib/libriemann_cpp.a \
           --no-whole-archive \
           -o /tmp/riemann_merged.o \
    && rm ${PKG_PREFIX}/lib/libriemann_cpp.a \
    && ar rcs ${PKG_PREFIX}/lib/libriemann_cpp.a /tmp/riemann_merged.o \
    && rm /tmp/riemann_merged.o

# Write DEBIAN/control
RUN INSTALLED_SIZE=$(du -sk ${DEB_ROOT}/usr | awk '{print $1}') && \
    mkdir -p ${DEB_ROOT}/DEBIAN && \
    cat > ${DEB_ROOT}/DEBIAN/control <<EOF
Package: ${PKG_NAME}
Version: ${VERSION}
Section: libdevel
Priority: optional
Architecture: amd64
Installed-Size: ${INSTALLED_SIZE}
Depends: libgomp1, libstdc++6
Maintainer: RiemannEngine <https://github.com/QuiteLiterallyConnor/RiemannEngine>
Description: RiemannEngine C++ development library
 Built for CUDA cu${CUDA_VERSION}.  Installs to
 /usr/local/RiemannEngine/${INSTALL_SUBDIR}; add that directory to
 CMAKE_PREFIX_PATH.
 .
 A C++/LibTorch sparse tensor library (sparse convolutions, adaptive
 voxelization, Gaussian splats) with CUDA acceleration.  Installs the static library, public headers,
 and CMake package config files so that downstream projects can consume the
 library via find_package(RiemannEngine CONFIG).
 .
 LibTorch is NOT bundled.  Install it separately and point CMake at it:
   cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch ..
EOF

# Write DEBIAN/postinst (update linker cache)
RUN cat > ${DEB_ROOT}/DEBIAN/postinst <<'EOF'
#!/bin/sh
set -e
ldconfig
EOF
RUN chmod 0755 ${DEB_ROOT}/DEBIAN/postinst

# Set correct permissions on all installed files
RUN find ${DEB_ROOT} -type d -exec chmod 755 {} \; && \
    find ${DEB_ROOT} -type f ! -name postinst -exec chmod 644 {} \;

# Build the .deb and the matching SDK tarball (RiemannEngine/<ver>_cu<X>/...)
RUN bash -c 'source /opt/cuda-profile.sh && \
    mkdir -p /output && \
    dpkg-deb --build --root-owner-group ${DEB_ROOT} \
        /output/libriemann-cpp-dev_${VERSION}_cu${CUDA_VERSION}_ubuntu${UBUNTU_VERSION}_amd64.deb && \
    tar -czf /output/RiemannEngine-${VERSION}-cu${CUDA_VERSION}-ubuntu${UBUNTU_VERSION}-x64.tar.gz \
        --owner=0 --group=0 -C ${DEB_ROOT}/usr/local RiemannEngine'

# ---------------------------------------------------------------------------
# Final stage — tiny image that holds only the .deb
# ---------------------------------------------------------------------------
FROM scratch AS artifact
COPY --from=builder /output /
