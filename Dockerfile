FROM python:3.8-slim-buster

WORKDIR /brainlit_dir

RUN apt-get update && apt-get install -y gcc libgl1-mesa-glx libxrender1
RUN apt install -y wget bzip2

RUN wget https://files.ilastik.org/ilastik-1.4.0-Linux.tar.bz2
RUN tar xjf ilastik-1.4.0-Linux.tar.bz2

COPY . .
RUN pip install -e .

RUN mkdir /root/.cloudvolume/
RUN mkdir /root/.cloudvolume/secrets/