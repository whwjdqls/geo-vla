#!/bin/bash


python ./examples/robocasa/eval_robocasa.py \
--policy.config=pi05_robocasa_pt --policy.dir=/scratch2/whwjdqls99/pi/pi05_robocasa_pt/base/30000 \
--args.out_dir /scratch2/whwjdqls99/pi/pi05_robocasa_pt/base/30000/eval_out \
--args.save-videos


#  --policy.dir /scratch2/whwjdqls99/pi/pi05_robocasa_pt/base/30000 --args.env-names CloseDoubleDoor --args.num-episodes-per-env 1 --args.num-steps-wait 10 --args.max-episode-steps 50 --args.save-videos --args.save-failed-videos --args.overwrite --args.out-dir /scratch2/whwjdqls99/pi/pi05_robocasa_pt/base/30000/eval_out_debug