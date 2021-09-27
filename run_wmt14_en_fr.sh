MAXTOKENS=3200
FRQ=8

# users' code dir
UDIR="deepnmt"

# translation task
TASK="adv_translation"

# model arch
ARCH="adv_transformer_wmt_en_de"
# Transformer Large
#ARCH="adv_transformer_vaswani_wmt_en_de_big"
	 
MODEL_PATH="/mnt/data/admin-data/models/wmt14-en-fr-admin-60-12l"
LOG="/mnt/data/admin-data/logs/wmt14-en-fr-admin-60-12l.log"
INITF="/mnt/duck/data/admin-data/init_files/wmt14-en-fr-admin-60-12l.txt"
DATA_PATH="/mnt/duck/data/admin-data/data/wmt14_en_fr_joined_dict/"
CKP="/mnt/duck/data/admin-data/models/wmt14-en-fr-gadmin-60-12l/checkpoint_last.pt"

# number of encoder layers
ELAYER=60
# number of decoder layers
DLAYER=12
# adv 0 denotes standard training
# adv 1 denotes vat
# adv 2 reg
# disable adv training in the release, just set AOPT=0
AOPT=0

# learning rate
LR=0.001

# Epoch

EPOCH=50


# Generate scalars using only one gpu

CUDA_VISIBLE_DEVICES=0 fairseq-train ${DATA_PATH} \
       --arch ${ARCH} --task ${TASK} --adv-opt 0 --share-all-embeddings --optimizer radam \
       --adam-betas '(0.9, 0.98)' --clip-norm 0.0 --lr-scheduler inverse_sqrt \
       --warmup-init-lr 1e-07 --warmup-updates 8000 --lr ${LR} --min-lr 1e-09 \
       --dropout 0.1 --attention-dropout 0.1 --relu-dropout 0.1 --weight-decay 0.0 \
       --criterion adv_label_smoothed_cross_entropy --label-smoothing 0.1 \
       --max-tokens $MAXTOKENS --update-freq $FRQ \
       --fp16 --fp16-scale-window 256 --threshold-loss-scale 0.03125 \
       --seed 1111 --restore-file x.pt --max-epoch ${EPOCH} --save-dir ${MODEL_PATH} \
       --encoder-layers ${ELAYER} --decoder-layers ${DLAYER} \
       --user-dir ${UDIR} --admin-init-type adaptive-profiling \
       --admin-init-path ${INITF} \
       --log-format simple --log-interval 100 | tee ${LOG}

# multi gpu training
GPUS=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
CUDA_VISIBLE_DEVICES=$GPUS fairseq-train ${DATA_PATH} \
       --arch ${ARCH} --task ${TASK} --adv-opt ${AOPT} --share-all-embeddings --optimizer radam \
       --adam-betas "(0.9,0.98)" --clip-norm 0.0 --lr-scheduler inverse_sqrt \
       --warmup-init-lr 1e-07 --warmup-updates 8000 --lr ${LR} --min-lr 1e-09 \
       --dropout 0.2 --attention-dropout 0.1 --relu-dropout 0.1 --weight-decay 0.0 \
       --criterion adv_label_smoothed_cross_entropy --label-smoothing 0.1 \
       --max-tokens $MAXTOKENS --update-freq $FRQ \
       --fp16 --fp16-scale-window 256 --threshold-loss-scale 0.03125 \
       --ddp-backend=no_c10d \
       --seed 1111 --restore-file ${CKP} --max-epoch ${EPOCH} --save-dir ${MODEL_PATH} \
       --encoder-layers ${ELAYER} --decoder-layers ${DLAYER} \
       --admin-init-path ${INITF} \
       --user-dir ${UDIR} --admin-init-type adaptive --log-format simple --log-interval 100 | tee ${LOG}
