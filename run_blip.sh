#!/bin/bash

# Navigate to model-store directory
cd ./model-store

# Remove existing blip2.mar file
rm blip2.mar
cd ..
# Go back to the project root directory
#cd /home/user01/projects/blip-serve

## Delete the existing model from the server
#curl -X DELETE http://localhost:8081/models/blip2

# Archive the new model
torch-model-archiver --model-name blip2 --version 1.0 --handler ./serve/handler.py --export-path ./model-store

# Deploy the new model to the server
curl -X POST "http://localhost:8081/models?url=blip2.mar&synchronous=true&min_worker=1"

# Update the model worker configuration
curl -X PUT "http://localhost:8081/models/blip2?synchronous=true&min_worker=1"
