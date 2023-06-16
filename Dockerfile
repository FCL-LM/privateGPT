FROM python:3.10
WORKDIR /privategpt

COPY . .

RUN pip install -r requirements.txt 

ENTRYPOINT [ "/privategpt/entrypoint.sh" ]