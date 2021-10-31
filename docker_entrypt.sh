#!/bin/bash
mkdir -p saved_models
/usr/bin/python3.6 data_tools/prepare_covid_data.py
/usr/bin/python3.6 data_tools/prepare_data.py
/usr/bin/python3.6 tools/transfer.py --cpu
/usr/bin/python3.6 tools/trainer.py \
    --cpu \
    --mode train \
    --freeze \
    --checkpoint models/CovidAID_transfered.pth.tar \
    --bs 16 \
    --save saved_models
# python tools/trainer.py \
#     --mode train \
#     --checkpoint <PATH_TO_BEST_MOMDEL> \
#     --bs 8 \
#     --save saved_models
# python tools/trainer.py \
#     --mode test \
#     --checkpoint <PATH_TO_BEST_MODEL> \
#     --cm_path plots/cm_best \
#     --roc_path plots/roc_best
# python tools/inference.py \
#     --img_dir <IMG_DIR> \
#     --checkpoint <BEST_MODEL_PTH>
