#!/bin/bash

model_name="cm3"
task="humaneval_generative"

#model_name_rep=`echo $model_name | sed -r 's/\//-/g'`
model_name_rep="${model_name}_6-2"

echo $model_name_rep

export TOKENIZERS_PARALLELISM=false

python -u main.py \
  --model cm3 \
  --tasks $task \
  $@ \
  | tee expts/${model_name_rep}_${task}.out
