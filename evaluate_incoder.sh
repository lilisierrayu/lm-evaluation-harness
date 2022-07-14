#!/bin/bash

model_name=$1
shift


#model_name_rep=`echo $model_name | sed -r 's/\//-/g'`
model_name_rep=`basename $model_name`

echo $model_name_rep

task=$1
if [ -z $task ]
then
  task="humaneval_generative"
else
  shift
fi

export TOKENIZERS_PARALLELISM=false

python -u main.py \
  --model incoder \
  --model_args pretrained=${model_name} \
  --device 0 \
  --batch_size 10 \
  --tasks $task \
  $@ \
  | tee expts/${model_name_rep}_${task}.out
