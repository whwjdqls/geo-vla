#!/bin/bash


torchrun --standalone --nnodes=1 --nproc_per_node=8 scripts/train_pytorch.py pi05_ours_low_mem_finetune_openvla_libero --exp_name debug --batch-size 128



python scripts/serve_policy.py policy:checkpoint --policy.config=pi05_ours_low_mem_finetune  --policy.dir=/scratch2/whwjdqls99/pi/pi05_ours_low_mem_finetune/train_pi_libero_128/30000

python /home/whwjdqls99/openpi/examples/libero/eval_libero_all.py --args.host 10.1.1.46 --args.out_dir /scratch2/whwjdqls99/pi/pi05_ours_low_mem_finetune/train_pi_libero_128/30000/eval_outputs --args.task_suite_name libero_goal

torchrun --standalone --nnodes=1 --nproc_per_node=4 scripts/train_pytorch.py pi05_ours_low_mem_finetune_openvla_libero_pt --exp_name pt_0.1 --batch-size 128 --aux-loss-weight 0.1 --resume 

torchrun --standalone --nnodes=1 --nproc_per_node=4 /home/whwjdqls99/openpi/scripts/train_pytorch.py pi05_ours_low_mem_finetune_openvla_libero_depth_latent --exp_name depth_latent_1.0_v2 --batch-size 128 