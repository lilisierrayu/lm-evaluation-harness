#!/bin/bash

model_name=$1
task=$2
shift
shift

#model_name_rep=`echo $model_name | sed -r 's/\//-/g'`
model_name_rep=`basename $model_name`

echo $model_name_rep

if [ -z $task ]
then
  task="incoder_python"
fi

python -u main.py \
  --model gpt2 \
  --model_args pretrained=${model_name} \
  --device cuda:0 \
  --batch_size 1 \
  --tasks $task \
  $@ \
  | tee expts/${model_name_rep}_${task}.out
