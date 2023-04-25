# syntax=docker/dockerfile:1

FROM python:3.8-slim-buster

COPY . .
RUN pip3 install -e .