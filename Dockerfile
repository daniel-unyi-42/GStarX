FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-runtime

RUN apt-get update -y && apt-get install -y --no-install-recommends \
    git \
    wget \
    tmux \
    vim \
    htop \
    openssh-server \
    zip \
	unzip \
	build-essential

RUN rm -rf /var/lib/apt/lists/*

RUN pip install torch_scatter==2.0.9 torch_sparse==0.6.13 torch_cluster==1.6.0 torch_spline_conv==1.2.1 -f https://data.pyg.org/whl/torch-1.10.0+cu113.html torch_geometric==1.7.2

RUN pip install debugpy pytest tensorboardX matplotlib seaborn pandas

RUN pip install numpy==1.20.3 \
hydra-core==1.1.1 \
matplotlib==3.4.3 \
networkx==2.6.3 \
seaborn==0.11.2 \
scikit-learn==0.24.2 \
scipy==1.6.3 \
pytz==2021.1 \
pyyaml==5.4.1 \
tqdm==4.53.0 \
dive-into-graphs==0.1.2

RUN useradd -m -s /bin/bash daniel

WORKDIR /home/daniel

USER daniel
