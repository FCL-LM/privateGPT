FROM python:3.10
WORKDIR /privategpt

COPY . .

RUN pip install -r requirements.txt 

HEALTHCHECK CMD curl --fail http://localhost:9000/health || exit 1
ENTRYPOINT [ "/privategpt/entrypoint.sh" ]