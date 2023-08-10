
FROM nvcr.io/nvidia/tensorrt:22.08-py3
# FROM baohuynhbk/tensorrt-ubuntu18.04-cuda11
# FROM python:3.8
# FROM nvcr.io/nvidia/l4t-jetpack:r35.1.0
# Docker images can be inherited from other images. Therefore, instead of creating our own base image, we’ll use the 
# official Python image that already has all the tools and packages that we need to run a Python application.
# can also use hub.docker.com to search for existing official images 

WORKDIR /app

# add python files to container
ADD 1file_path_indexer.py .
ADD 2video_to_frame_splitter.py .
ADD 3frames_to_process.py .
ADD 4make_model_json_files.py .
ADD 5analyser.py . 
ADD AAframe_fetcher.py .
ADD Aflaskapp.py .
ADD common.py .
ADD network.py .
ADD onnx_to_tensorrt.py .
ADD rn50_data_processing.py .
ADD data_processing.py .
ADD downloader.py .
ADD coco_labels.txt .


# RUN apt-get -qq update && apt-get install build-essential

# need to make a file containing all the dependencies to this working directory
COPY requirements.txt .

# install all packages; install all modules into the image; install 3rd party library; dependencies
RUN apt-get update && apt install -y python3-opencv
RUN pip3 install -r requirements.txt

#COPY . .
#RUN pip3 install 'C:\Users\arzina.ismaili\OneDrive - Chess Dynamics Ltd\Desktop\TensorRT-8.2.0.6\python\tensorrt-8.2.0.6-cp38-none-win_amd64.whl'

# copy entire app to directory; here the {. = /app}; i.e. copy app from /app to /app
COPY . ./apptwo 

# specifiy the entry command when we start the container
#CMD ["python", "./5analyser.py"]
