FROM nvcr.io/nvidia/pytorch:21.09-py3

USER root
WORKDIR /code

ENV DEBIAN_FRONTEND=noninteractive TIME_ZONE=Asia/Shanghai LANG=C.UTF-8 LC_ALL=C.UTF-8 TERM=xterm-256color

# 修改apt源
RUN mv /etc/apt/sources.list.d/ /etc/apt/sources.list.d.bk && apt update && apt install apt-transport-https ca-certificates
COPY ./sources.list /etc/apt/sources.list

# 安装一些必备工具
RUN apt-get update && apt install -y vim wget curl sudo inetutils-ping net-tools zip unzip git openssh-server openssh-client build-essential tzdata lintian dh-make devscripts pkg-config \
    libgtk2.0-dev libgtk-3-dev libavcodec-dev libavformat-dev libjpeg-dev libswscale-dev libtiff5-dev \
    && apt install -y --fix-missing cmake clangd-12 ffmpeg autoconf automake libgl1-mesa-dev libsm6 libxext6 git ninja-build libglib2.0-dev \
    && ln -snf /usr/share/zoneinfo/$TIME_ZONE /etc/localtime && echo $TIME_ZONE > /etc/timezone && dpkg-reconfigure -f noninteractive tzdata && apt-get clean && rm -rf /var/lib/apt/lists/ \
        && update-alternatives --install /usr/bin/clangd clangd /usr/bin/clangd-12 100 && echo 'export LANG="C.UTF-8"' >> /etc/profile && /bin/bash -c "source /etc/profile"


# mmcv cross CUDA architecture support
ENV MMCV_WITH_OPS=1 \
    TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6+PTX" \
    TORCH_NVCC_FLAGS="-Xfatbin -compress-all" 
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple && pip install pip -U
RUN pip install openmim && mim install  mmcv-full==1.3.14 && pip install  mmdet==2.14.0 && pip install  mmsegmentation==0.14.1 && rm -rf /tmp/*

RUN pip install pandas==1.3.5 && pip install scikit-image==0.18.3 && pip install lyft_dataset_sdk && pip install  networkx==2.2\
    && pip install numba==0.48.0 --ignore-installed llvmlite \
    && pip install numpy==1.19.5 \
    && pip install nuscenes-devkit==1.1.9 \
    && pip install plyfile tensorboard fvcore \
    && pip install trimesh==2.35.39\
    && pip install torchmetrics==0.11.4 && pip install opencv-python==4.2.0.34 && rm -rf /tmp/*

# COPY track-2.tar /code/
# RUN cd /code/ && tar -xvf track-2.tar && rm track-2.tar && cd track-2/BEVerse && python setup.py develop
# wheel built specifically for this image and beverse config 
COPY mmdet3d-0.17.2-cp38-cp38-linux_x86_64.whl BEVerse.tar /code/
RUN pip install mmdet3d-0.17.2-cp38-cp38-linux_x86_64.whl && rm mmdet3d-0.17.2-cp38-cp38-linux_x86_64.whl && tar -xvf BEVerse.tar && rm BEVerse.tar
ENV PYTHONPATH=/code/BEVerse

RUN mkdir /var/run/sshd \
    && echo "root:123456" | chpasswd \
    && sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config \
    && sed -i '/^session\s\+required\s\+pam_loginuid.so/s/^/#/' /etc/pam.d/sshd \
    && sed -i 's/UsePAM yes/UsePAM no/g' /etc/ssh/sshd_config\
    && mkdir -p /root/.ssh && chown root.root /root && chmod 700 /root/.ssh
EXPOSE 22 8888 8080
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT [ "/entrypoint.sh" ]
