FROM python:3.10
WORKDIR /privategpt

COPY requirements.txt .
RUN pip install -r requirements.txt 

COPY . .

HEALTHCHECK CMD curl --fail http://localhost:9000/health || exit 1
ENTRYPOINT [ "/privategpt/entrypoint.sh" ]