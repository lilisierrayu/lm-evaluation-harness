#!/bin/bash

model_name="cm3"

#model_name_rep=`echo $model_name | sed -r 's/\//-/g'`
model_name_rep="${model_name}"

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
  --model cm3 \
  --tasks $task \
  $@ \
  | tee expts/${model_name_rep}_${task}.out
