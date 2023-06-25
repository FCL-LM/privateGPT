#!/bin/bash
printenv >> /etc/environment
python /privategpt/fetch_models.py
[ $? -ne 0 ] && echo "Error in fetch_models.py" && exit 1

python /privategpt/ingest.py
[ $? -ne 0 ] && echo "Error in ingest.py" && exit 1

python /privategpt/flaskAPP.py