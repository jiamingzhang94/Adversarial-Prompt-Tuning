
# custom config
ROOT="/vhome/user/dataset"
TRAINER=AdvPT
# oxford_flowers, oxford_pets, imagenet, food101, sun397, dtd, eurosat, ucf101
DATASET=oxford_flowers
# rn50, vit_b16, vit_l14
CFG=vit_l14 # config file
CTP=end  # class token position (end or middle)
NCTX=32  # number of context tokens
#SHOTS=16  # number of shots (1, 2, 4, 8, 16)
CSC=False  # class-specific context (False or True)

D=$ROOT
SEED=1

DIR=/share/test/user/share1/new/${DATASET}/${TRAINER}/${CFG}/adv
echo "--------------------------------------------------------------------------------------"
python train.py \
--root ${D} \
--adv-training \
--seed ${SEED} \
--trainer ${TRAINER} \
--dataset-config-file configs/datasets/${DATASET}.yaml \
--config-file configs/trainers/${TRAINER}/${CFG}.yaml \
--output-dir ${DIR} \
--model-dir ${DIR} \
--adv-training \
TRAINER.ADV.N_CTX ${NCTX} \
TRAINER.ADV.CLASS_TOKEN_POSITION ${CTP} \
TRAINER.ADV.CSC ${CSC}



echo "--------------------------------------------------------------------------------------"
echo "zero shot"
TRAINER=ZeroshotCLIP
python train.py \
--root ${D} \
--trainer ${TRAINER} \
--dataset-config-file configs/datasets/${DATASET}.yaml \
--config-file configs/trainers/ADV/${CFG}.yaml \
--output-dir output/${TRAINER}/${CFG}/${DATASET} \
--eval-only