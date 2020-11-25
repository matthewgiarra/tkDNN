#!/bin/bash

docker run --rm -it -v $(pwd):/workspace --gpus all tkdnn:build /bin/bash -c "mkdir -p /workspace/build && cd /workspace/build && cmake .. && make trimmer -j12"