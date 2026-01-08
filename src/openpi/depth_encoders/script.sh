


python train_AE.py \
    --root /scratch2/whwjdqls99/libero/libero_hdfr_lerobot_datasets_only_depths \
    --meta-json-name depth_u16_png_metadata.json \
    --epochs 100 \
    --z-hw 16 \
    --z-ch 4 \
    --batch-size 128 \
    --num-workers 8 \
    --outdir ./runs/z_hw_16_z_ch4_temp1_new \
    --wandb --wandb-name z_hw_16_z_ch4_temp1_new