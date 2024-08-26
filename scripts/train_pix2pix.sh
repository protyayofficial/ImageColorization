#!/bin/bash

# Set default values for arguments
BATCH_SIZE=64
NUM_WORKERS=4
NUM_EPOCHS=100
VISUALIZE_INTERVAL=10
METRICS_INTERVAL=1
SEED=9
SAVE_DIR="experiments"
MODEL_NAME="pix2pix"
TRAIN_DATA_PATH="coco/test2017"
VAL_DATA_PATH="coco/val2017"

# Parse command-line arguments
while getopts "b:w:e:v:m:s:d:n:t:v:" opt; do
  case $opt in
    b) BATCH_SIZE=$OPTARG ;;
    w) NUM_WORKERS=$OPTARG ;;
    e) NUM_EPOCHS=$OPTARG ;;
    v) VISUALIZE_INTERVAL=$OPTARG ;;
    m) METRICS_INTERVAL=$OPTARG ;;
    s) SEED=$OPTARG ;;
    d) SAVE_DIR=$OPTARG ;;
    n) MODEL_NAME=$OPTARG ;;
    t) TRAIN_DATA_PATH=$OPTARG ;;
    v) VAL_DATA_PATH=$OPTARG ;;
    \?) echo "Invalid option -$OPTARG" >&2 ;;
  esac
done

# Run the training script with provided parameters
python train.py --batch_size $BATCH_SIZE --num_workers $NUM_WORKERS --num_epochs $NUM_EPOCHS --visualize_interval $VISUALIZE_INTERVAL --metrics_interval $METRICS_INTERVAL --seed $SEED --save_dir $SAVE_DIR --model_name $MODEL_NAME --train_data_path $TRAIN_DATA_PATH --val_data_path $VAL_DATA_PATH
