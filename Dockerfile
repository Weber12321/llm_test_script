# syntax=docker/dockerfile:1

# preparation for building image:

# $ eval $(ssh-agent)
# $ ssh-add ~/.ssh/name_of_key
# $ # (may need to input passphrase)

# ref: https://docs.docker.com/develop/develop-images/build_enhancements/#using-ssh-to-access-private-data-in-builds

# $ DOCKER_BUILDKIT=1 docker build -t expr_llm:eval .

FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND="noninteractive"

ARG DEPENDENCIES='python3 python3-pip automake bash ca-certificates \
                  g++ git libtool libleptonica-dev make pkg-config \
                  libpango1.0-dev libicu-dev libcairo2-dev libglib2.0-0 \
                  libsm6 libxext6 libxrender-dev libgl1-mesa-glx musl-dev'

ARG OPT_DEPENDENCIES='asciidoc docbook-xsl xsltproc wget unzip bc'

RUN sed -i 's/archive.ubuntu.com/tw.archive.ubuntu.com/g' /etc/apt/sources.list
RUN apt-get update \
    && apt upgrade -y \
    && apt-get install -y ${DEPENDENCIES} \
    && apt-get install -y --no-install-recommends ${OPT_DEPENDENCIES} \
    && apt-get clean \
    && apt-get autoremove

COPY ./requirements ./requirements

RUN --mount=type=ssh python3 -m pip install --no-cache-dir --upgrade pip \
    # && python3 -m pip install --no-cache-dir Cython==3.0.0 \
    && pip install --no-cache-dir -r ./requirements/base

RUN adduser --disabled-password systalkai

RUN mkdir -p /expr

WORKDIR /expr

COPY ./apps /expr/apps

RUN chown -R systalkai:systalkai /expr

WORKDIR /expr/apps

COPY ./scripts /expr/scripts

WORKDIR /expr/apps

USER systalkai