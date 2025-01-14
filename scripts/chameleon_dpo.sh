#!/usr/bin/env bash
#
# Copyright 2024 PKU-Alignment Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

MODEL_NAME_OR_PATH="PKU-Alignment/AA-chameleon-7b-base" # model path

TRAIN_DATASETS="/data/dataset" # dataset path
PT_NAME="aa-ti2ti-pretokenize-output.pt"
OUTPUT_DIR="../outputs/chameleon_dpo" # output dir

# For wandb online logging
export WANDB_API_KEY=${WANDB_API_KEY}

# Source the setup script
source ./setup.sh

# Execute deepspeed command
deepspeed \
    --master_port ${MASTER_PORT} \
    --module align_anything.trainers.text_image_to_text_image.dpo \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --train_datasets ${TRAIN_DATASETS} \
    --train_data_files ${PT_NAME} \
    --output_dir ${OUTPUT_DIR} \
    --train_template ANYTHING_TI2TI \
    --train_split 'train' \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 2 \
    --save_interval 2500 \
    --learning_rate 5e-7 \
    --epochs 3 \
    --lr_scheduler_type cosine