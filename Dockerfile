FROM python:3.10
WORKDIR /privategpt

RUN apt-get update && \
    apt-get -y install \
    poppler-utils \
    libsm6 \
    libxext6 \
    libgl1 \
    tesseract-ocr

COPY requirements.txt .
RUN pip install -r requirements.txt 
RUN python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'

COPY . .

HEALTHCHECK CMD curl --fail http://localhost:9000/health || exit 1
ENTRYPOINT [ "/privategpt/entrypoint.sh" ]