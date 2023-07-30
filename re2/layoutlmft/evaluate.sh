#!/bin/bash

for i in {10..2000..10}
do
  mkdir ../temp_eval/eval_$i
  CUDA_VISIBLE_DEVICES=3 python -m torch.distributed.launch --nproc_per_node=1 --master_port 47769 examples/run_xfun_re.py --model_name_or_path path/to/checkpoint --output_dir path/to/output_directory --do_eval --lang zh --max_steps 2500 --per_device_train_batch_size 2 --warmup_ratio 0.1 --fp16
  PID=$!
  wait $PID
done
