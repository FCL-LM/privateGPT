#!/bin/bash
/privategpt/fetch_models.py
[ $? -ne 0 ] && echo "Error in fetch_models.py" && exit 1

/privategpt/ingest.py
[ $? -ne 0 ] && echo "Error in ingest.py" && exit 1

/privategpt/flaskAPP.py