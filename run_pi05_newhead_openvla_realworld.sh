export HF_HUB_OFFLINE=1
export LEROBOT_VIDEO_BACKEND=pyav
export CUDA_VISIBLE_DEVICES=4,5,6,7

torchrun --standalone --nnodes=1 --nproc_per_node=4 scripts/train_pytorch.py \
    pi05_real_world_pt_v3_new_head \
    --exp_name pi05_ours_track_real_world_geo_pnp_doll_obstacle_0125 \
    --pytorch-weight_path /weka/jisookim/experiment/pi05/pi05_base_pytorch \
    --batch-size 32 \
    --no-wandb-enabled \
    --aux-loss-weight 1.0 \
    --save-interval 5000 \
    --checkpoint_base_dir /weka/jisookim/experiment/pi05 \
    --data.root_dir /weka/jisookim/dataset/real_world_lerobot/omy_f3m_geo_pnp_doll_obstacle_0125_pt \
    --num-train-steps 150000
