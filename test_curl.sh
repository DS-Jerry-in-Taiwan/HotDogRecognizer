#!/bin/bash

curl -X POST "http://127.0.0.1:8000/predict/" \
  -F "file=@/home/ubuntu/projects/HotDogRecognizer/data/hotdog/test/hotdog/1000.png"
