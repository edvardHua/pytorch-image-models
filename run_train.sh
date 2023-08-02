
### best result at  './output/train/20230320-162638-convnext_base_clip_laion2b-256/checkpoint-2.pth.tar', 0.88
# CUDA_VISIBLE_DEVICES=0 python train.py \
# --data-dir /home/wenqi.wang/datasets/abnormal_lighting/train_normal_abnormal \
# --train-split "training" \
# --val-split "validation" \
# --model convnext_base.clip_laion2b \
# --pretrained \
# --batch-size 32 \
# --validation-batch-size 16 \
# --epochs 30 \
# --num-classes 2 \
# --workers 16 \
# --opt adamw \
# --weight-decay 0.02 \
# --lr 1e-4 \
# --no-aug \
# --drop-path 0.1 \
# --use-multi-epochs-loader \
# --checkpoint-hist 5 \
# --eval-metric recall \

#--aa v0 \

##--experiment abnormallight3 \


# CUDA_VISIBLE_DEVICES=0 python train.py \
# --data-dir /home/wenqi.wang/datasets/abnormal_lighting/train_ds_0301 \
# --train-split "training" \
# --val-split "validation" \
# --model resnet50 \
# --pretrained \
# --batch-size 32 \
# --validation-batch-size 16 \
# --epochs 30 \
# --num-classes 3 \
# --workers 16 \
# --opt adamw \
# --weight-decay 0.02 \
# --lr 1e-3 \
# --no-aug \
# --drop-path 0.1 \
# --use-multi-epochs-loader \
# --checkpoint-hist 5 \
# --eval-metric recall \
#--log-wandb \


## --mixup 0.3 \
## --color-jitter 0.0 \
## --experiment abnormallight2 \
## --model resnet50 \
# --model-ema \
# --model-ema-decay 0.99 \
## --no-aug \
## --sched consine \
# --mixup 0.2 \
# --mixup-prob 0.5 \
# --bce-loss \
## --cutmix 0.3 \
## --model swinv2_base_window12_192.ms_in22k
##--experiment abnormallight2_resnet50_ce_cutmix_cosine \ 
## --model convnext_base.clip_laion2b

CUDA_VISIBLE_DEVICES=1 python3 train_edz.py \
--data-dir /home/zihua/data_warehouse/low_quality/ablight_screened_0719_ds \
--output /home/zihua/data_warehouse/low_quality/ablight_model \
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
--experiment resnet50_ablight_screened_0719

