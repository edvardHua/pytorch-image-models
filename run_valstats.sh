# --model convnext_base.clip_laiona_augreg_320 \
# --checkpoint ./output/train/abnormallight2_convnext320_ce_cutmix_cosine/model_best.pth.tar

# --model convnext_base.clip_laiona_augreg_320 \
# --checkpoint ./output/train/abnormallight2_convnext320_ce_cutmix_cosine/model_best.pth.tar

CUDA_VISIBLE_DEVICES=0 python3 validate_stats.py \
--data-dir /mnt/zihua.zeng/train_ds_0322 \
--split "validation" \
--batch-size 16 \
--num-classes 2 \
--workers 16 \
--device "cpu" \
--model convnext_base.clip_laion2b \
--checkpoint ./output/train/20230322-074458-convnext_base_clip_laion2b-256//model_best.pth.tar


