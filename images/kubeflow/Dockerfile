ARG IMAGE_TYPE="cpu"
FROM gcr.io/kubeflow-images-public/tensorflow-1.14.0-notebook-${IMAGE_TYPE}:v0.7.0

USER root

# Install basic dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ca-certificates bash-completion tar less gettext-base \
        python-pip python-setuptools build-essential python-dev \
        python3-pip python3-wheel libgl1-mesa-glx && \
    rm -rf /var/lib/apt/lists/*

ENV SHELL /bin/bash
COPY bashrc /etc/bash.bashrc
RUN echo "set background=dark" >> /etc/vim/vimrc.local

# Install latest KFP SDK & Kale & JupyterLab Extension
RUN pip3 install --upgrade pip && \
    pip3 install --upgrade "jupyterlab<2.0.0" && \
    pip3 install https://storage.googleapis.com/ml-pipeline/release/latest/kfp.tar.gz --upgrade && \
    pip3 install kubeflow-kale==0.5.0 && \
    jupyter labextension install kubeflow-kale-labextension


RUN pip install alibi==0.3.2

RUN pip install -U tensorflow && \
    pip install alibi-detect

RUN pip install minio

# Seldon currently runs an old version of sklearn
RUN pip install scikit-learn==0.20.3

# Need this untill upgrade of explainer wrapper as 0.3.2 is incompatible
RUN pip install dill==0.3.1.1

RUN pip install kfserving

WORKDIR /home
COPY swagger-codegen/out/deploy deploy
WORKDIR /home/deploy
RUN python setup.py install

RUN echo "jovyan ALL=(ALL:ALL) NOPASSWD:ALL" > /etc/sudoers.d/jovyan

WORKDIR /home/models
COPY samples samples
RUN chown -R jovyan /home/models 

WORKDIR /home/jovyan
USER jovyan

CMD ["sh", "-c", \
     "jupyter lab --notebook-dir=/home/jovyan --ip=0.0.0.0 --no-browser \
      --allow-root --port=8888 --LabApp.token='' --LabApp.password='' \
      --LabApp.allow_origin='*' --LabApp.base_url=${NB_PREFIX}"]

