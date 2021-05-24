#!/usr/bin/env bash

# run the docker image for hosting jupyter
docker run \
	--runtime=nvidia \
	--rm -d -p 9240:9240 \
	-v ~/Code/image-triplet-representations:/home/ \
	-v ~/Code/face-morphology/data/datasets/celeba:/home/data/celeba \
	-w /home/ \
	--name image-triplet image-triplet:0.1 \
	docker/run_jupyter.sh
