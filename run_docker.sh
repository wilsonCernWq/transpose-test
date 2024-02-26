#!/bin/bash

docker run -it --rm --privileged --device /dev/dri \
    -v "$PWD":/workspace \
    -v /dev/dri/by-path:/dev/dri/by-path \
    -v /usr/local:/usr/local \
    -w /workspace \
    new_tsyclnn
