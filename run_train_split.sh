CUDA_VISIBLE_DEVICES=1 python3 train_edz.py \
--data-dir /home/zihua/data_warehouse/low_quality/split_image_ds \
--output /home/zihua/data_warehouse/ablight_model \
--train-split "training" \
--val-split "validation" \
--model resnet50 \
--pretrained \
--batch-size 32 \
--input-size 3 368 368 \
--validation-batch-size 16 \
--epochs 30 \
--num-classes 2 \
--workers 16 \
--opt adamw \
--weight-decay 0.02 \
--sched cosine \
--lr 1e-4 \
--color-jitter 0.0 \
--drop-path 0.1 \
--use-multi-epochs-loader \
--checkpoint-hist 5 \
--eval-metric recall \
--experiment resnet50_split_image

##--log-wandb \
