FROM python:slim
RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && apt-get purge -y --auto-remove \
    && rm -rf /var/lib/apt/lists/*
# RUN apk update && apk upgrade && \
#     apk add --no-cache bash git openssh
# specify working directory
WORKDIR /usr/src/app

# download brainlit
RUN git clone https://github.com/neurodata/brainlit.git
RUN cd brainlit && \
    pip install -e .

# create credentials file for aws
RUN chmod +x .aws.sh && \ 
    source .aws.sh

