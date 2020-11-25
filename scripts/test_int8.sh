#!/bin/bash
DATA_DIR=$HOME/data

docker run --rm -it -v $(pwd):/workspace -v $DATA_DIR:/data --gpus all tkdnn:build ./build/trimmer rt/yolo4_int8.rt /data/videos/gta_lobby_416.mp4 y 80 16 0 0.30 config/trimmer_config.json