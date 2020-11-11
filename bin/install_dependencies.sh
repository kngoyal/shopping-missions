#!/bin/bash

aws s3 cp s3://nlp-use-cases/bootstrap/requirements.txt .
sudo python3 -m pip install -r requirements.txt