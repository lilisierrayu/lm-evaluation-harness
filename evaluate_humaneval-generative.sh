#!/bin/bash

model_name=$1
task="humaneval_generative"
shift

#model_name_rep=`echo $model_name | sed -r 's/\//-/g'`
model_name_rep=`basename $model_name`

echo $model_name_rep

export TOKENIZERS_PARALLELISM=false

python -u main.py \
  --model incoder \
  --model_args pretrained=${model_name} \
  --device 0 \
  --batch_size 1 \
  --tasks $task \
  $@ \
  | tee expts/${model_name_rep}_${task}.out
