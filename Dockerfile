# from python 3.7
FROM python:3.7

RUN apt-get update --fix-missing
RUN apt-get install -y --no-install-recommends build-essential
RUN apt-get install --yes libsndfile1
RUN apt-get update --fix-missing && apt-get install libsndfile1 ffmpeg libsox-fmt-all sox -y

WORKDIR /src
COPY . . 

RUN pip3 install --no-cache-dir Cython
RUN pip3 install --upgrade numpy
RUN pip3 install -e . 
RUN pip3 install --upgrade numpy

ENTRYPOINT ["python", "predict.py"]