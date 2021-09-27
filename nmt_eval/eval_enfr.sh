#!/usr/bin/env bash
GPUS=0
DATA_PATH="/mnt/duck/data/admin-data/data/wmt14_en_fr_joined_dict/"

echo "Model path" $SAVEDIR

if [[ $# -lt 2 ]]; then
  echo "bash.sh <model_path> <gpu> <init_path> <unperbound> <count>"
  exit 1
fi


GPUDEV=${2:-0}
SAVEDIR=${1}
INIT_PATH=${3}
UPPER_BOUND=${4:-50}
CP_POINT_NUM=${5:-10}
MODELDIR=$SAVEDIR/model_${UPPER_BOUND}_${CP_POINT_NUM}.pt
echo "MODELIR"

cp ${INIT_PATH} ./profile.init

echo "python average_checkpoints.py --inputs $SAVEDIR --num-epoch-checkpoints ${CP_POINT_NUM}  --output $MODELDIR --checkpoint-upper-bound $UPPER_BOUND --admin-init-path ${INIT_PATH}" 
if [ -f $MODELDIR  ]; then
    echo $MODELDIR "already exists"
else
    echo "Start averaging model"
    python average_checkpoints.py --inputs $SAVEDIR --num-epoch-checkpoints ${CP_POINT_NUM}  --output $MODELDIR --checkpoint-upper-bound $UPPER_BOUND --admin-init-path ${INIT_PATH} # | grep 'Finish'
    echo "End averaging model"
fi


BFOUT="${SAVEDIR}/test_${UPPER_BOUND}_${CP_POINT_NUM}_bleu.txt"
CUDA_VISIBLE_DEVICES=$GPUDEV fairseq-generate ${DATA_PATH} \
                    --path $MODELDIR \
                    --batch-size 256 --beam 10 --lenpen 1.0 --remove-bpe \
                    > ${BFOUT}

echo "BLEU"
tail -2 ${BFOUT}
