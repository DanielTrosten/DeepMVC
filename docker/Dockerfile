FROM pytorch/pytorch:latest
#FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-devel

WORKDIR /storage

RUN apt-get update
#RUN apt-get install -y \
#    libgl1-mesa-glx \
#    libglib2.0-0

RUN pip uninstall -y torch
RUN pip uninstall -y torchtext

COPY requirements.txt .
RUN pip install -r requirements.txt
RUN rm requirements.txt

RUN conda install -y -c conda-forge faiss
