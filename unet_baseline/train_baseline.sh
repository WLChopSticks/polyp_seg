#!/usr/bin/env bash
# python train.py --model_name unet -batch_size 4 --img_size 256 --gpu_order 0 --lr 1e-4 --num_epoch 70  --loss ce-dice --lr_policy StepLR --log_name ce-dice-Step.log --params_name ce-dice-Step_params.pkl --resume 0
python train.py --data_root /home/jiaxin/MICCAI2020/data/CVC-912  --train_csv  /home/jiaxin/MICCAI2020/data/csv/5_folds/train_data_split1.csv --test_csv /home/jiaxin/MICCAI2020/data/csv/5_folds/val_data_split1.csv --batch_size 4  --num_epoch 50 --lr_policy StepLR --fold_num 1

python train.py --data_root /home/jiaxin/MICCAI2020/data/CVC-912  --train_csv  /home/jiaxin/MICCAI2020/data/csv/5_folds/train_data_split2.csv --test_csv /home/jiaxin/MICCAI2020/data/csv/5_folds/val_data_split2.csv --batch_size 4  --num_epoch 50 --lr_policy StepLR --fold_num 2

python train.py --data_root /home/jiaxin/MICCAI2020/data/CVC-912  --train_csv  /home/jiaxin/MICCAI2020/data/csv/5_folds/train_data_split3.csv --test_csv /home/jiaxin/MICCAI2020/data/csv/5_folds/val_data_split3.csv --batch_size 4  --num_epoch 50 --lr_policy StepLR --fold_num 3

python train.py --data_root /home/jiaxin/MICCAI2020/data/CVC-912  --train_csv  /home/jiaxin/MICCAI2020/data/csv/5_folds/train_data_split4.csv --test_csv /home/jiaxin/MICCAI2020/data/csv/5_folds/val_data_split4.csv --batch_size 4  --num_epoch 50 --lr_policy StepLR --fold_num 4

python train.py --data_root /home/jiaxin/MICCAI2020/data/CVC-912  --train_csv  /home/jiaxin/MICCAI2020/data/csv/5_folds/train_data_split5.csv --test_csv /home/jiaxin/MICCAI2020/data/csv/5_folds/val_data_split5.csv --batch_size 4  --num_epoch 50 --lr_policy StepLR --fold_num 5

