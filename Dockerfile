
FROM python:3.8-slim-buster

WORKDIR /brainlit_dir

RUN apt-get update && apt-get install -y gcc libgl1-mesa-glx libxrender1
COPY . .
RUN pip install -e .
