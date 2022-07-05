#!/bin/bash

model_name="cm3"
task="humaneval_generative"
shift

#model_name_rep=`echo $model_name | sed -r 's/\//-/g'`
model_name_rep="${model_name}_6-2"

echo $model_name_rep

python -u main.py \
  --model cm3 \
  --model_args generation_temperature=0.2 \
  --device 0 \
  --batch_size 1 \
  --tasks $task \
  $@ \
  | tee expts/${model_name_rep}_${task}.out
