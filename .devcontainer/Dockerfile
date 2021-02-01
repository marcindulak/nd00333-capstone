# curl -L https:///mcr.microsoft.com/v2/vscode/devcontainers/miniconda/tags/list
FROM mcr.microsoft.com/vscode/devcontainers/miniconda:0.154.1-3

# Update args in docker-compose.yaml to set the UID/GID of the "vscode" user.
ARG USER_UID=1000
ARG USER_GID=$USER_UID
RUN if [ "$USER_GID" != "1000" ] || [ "$USER_UID" != "1000" ]; then \
        groupmod --gid $USER_GID vscode \
        && usermod --uid $USER_UID --gid $USER_GID vscode \
        && chown -R $USER_UID:$USER_GID /home/vscode; \
    fi

# [Optional] Uncomment this section to install additional OS packages.
RUN curl -sO https://packages.microsoft.com/repos/azure-cli/pool/main/a/azure-cli/azure-cli_2.18.0-1~buster_all.deb \
    && dpkg -i azure-cli_2.18.0-1~buster_all.deb \
    && az --version \
    && rm -f azure-cli_2.18.0-1~buster_all.deb
# knack.util.CLIError: Please run 'az login' to setup account.
RUN az extension add --name azure-cli-ml --version 1.21.0 --debug \
    && az --version
#RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
#    && apt-get -y install --no-install-recommends time \
#    && rm -rf /var/lib/apt/lists/*
RUN rm -rf /var/lib/apt/lists/*

# Install Node.js
ARG INSTALL_NODE="true"
ARG NODE_VERSION="lts/fermium"
RUN if [ "${INSTALL_NODE}" = "true" ]; then su vscode -c "source /usr/local/share/nvm/nvm.sh && nvm install ${NODE_VERSION} 2>&1"; fi

# Give the vscode the ownership of /opt/conda and /tmp/conda-tmp
RUN chown -R $USER_UID:$USER_GID /opt/conda

# Use vscode user
USER vscode

# Pin coda to use only conda-forge channel
RUN conda config --add channels conda-forge \
    && conda config --remove channels defaults \
    && conda config --show channels

# Install a specific conda itself
RUN conda install -y -c defaults conda=4.9.2 \
    && conda clean --all

# [Optional] Uncomment to install a different version of Python than the default
RUN conda install -y -c conda-forge python=3.7.9 \
    && conda clean --all

# Install jupyter notebook
RUN conda install -y -c conda-forge notebook=6.1 \
    && conda clean --all \
    && jupyter kernelspec list

# Copy environment.yml (if found) to a temp location so we update the environment. Also
# copy "noop.txt" so the COPY instruction does not fail if no environment.yml exists.
COPY environment.yml* .devcontainer/noop.txt /tmp/conda-tmp/
RUN sudo chown -R $USER_UID:$USER_GID /tmp/conda-tmp
RUN if [ -f "/tmp/conda-tmp/environment.yml" ]; then /opt/conda/bin/conda env update -n base -f /tmp/conda-tmp/environment.yml; fi \
    && rm -rf /tmp/conda-tmp \
    && conda clean --all \
    && pip cache purge

# Set the current workdir
WORKDIR /app
