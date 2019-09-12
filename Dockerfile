ARG BASE_CONTAINER=jupyter/tensorflow-notebook
FROM $BASE_CONTAINER

USER root

WORKDIR /ml-school

# install rennet
RUN git clone https://github.com/fraunhofer-iais/rennet.git \
    && pip install ./rennet[analysis,test] \
    && conda clean --all -f -y \
    && fix-permissions $CONDA_DIR \
    && fix-permissions /home/$NB_USER

# Switch back to jovyan to avoid accidental container runs as root
USER $NB_UID