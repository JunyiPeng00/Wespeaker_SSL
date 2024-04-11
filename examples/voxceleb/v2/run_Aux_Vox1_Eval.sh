#!/bin/bash
#SBATCH -J Improved_SSL_SPK-Encoder
#SBATCH -p gpu-a6000-kishin
#SBATCH --gres=gpu:4
#SBATCH -c 4

WORK_DIR=/home/jpeng/ntt/work/SPKID/examples/voxceleb/v2

source /home/jpeng/anaconda3/bin/activate /home/jpeng/anaconda3/envs/wespeaker

# which python
cd $WORK_DIR

. ./path.sh || exit 1


stage=3
stop_stage=3 #6

data=data
data_type="raw"  # shard/raw

adapter_dim=128

# config=conf/compare/wavlm_base_CorrPoolingDrop.yaml # wavlm_base_MHFA_LR
# config=conf/wavlm_base_MHFA_LR_Pooling_Group4_Conv2D_32_L4_Drop.yaml # wavlm_base_MHFA_LR
# config=conf/adapter/wavlm_base_SeqAdapter.yaml
config=conf/wavlm_base_MHFA_LR.yaml # wavlm_base_MHFA_LR
config=conf/CA_MHFA/wavlm_base_MHFA_LR_Pooling_Group4_Conv2D_32_L4.yaml


# exp_dir=exp/WavLM-BasePlus-FullFineTuning-MHFA-emb256-3s-LRS10-Epoch40-LM
# exp_dir=exp/CA-MHFA/WavLM-BasePlus-FullFineTuning-G-MHFA_Conv2D-emb256-3s-LRS10-Epoch40-Head64-L2
# exp_dir=exp/LargeModel/WavLM-Large-FullFineTuning-MHFA-Head32-emb256-3s-LRS10-Epoch20
# exp_dir=exp/WavLM-BasePlus-FullFineTuning-MHFA-Head32-emb256-3s-LRS10-Epoch40
# exp_dir=exp/CA-MHFA/WavLM-BasePlus-FullFineTuning-G-MHFA_Conv2D-emb256-3s-LRS10-Epoch40-Head32-L4
# exp_dir=exp/TMP_${port}
# exp_dir=exp/V2/WavLM-BasePlus-FullFineTuning-MHFA-emb256-3s-LRS10-Epoch30
# exp_dir=exp/V2/WavLM-BasePlus-FullFineTuning-MHFA-emb256-3s-LRS10-Epoch20-CNN_Learnable
# exp_dir=exp/V2/WavLM-BasePlus-FullFineTuning-MHFA-emb256-3s-LRS10-Epoch20-CNN_Learnable-speed
exp_dir=exp/LargeModel/WavLM-Large-FullFineTuning-MHFA-Head32-emb256-3s-LRS10-Epoch20-2
# exp_dir=exp/LargeModel/WavLM-Large-FullFineTuning-CA-MHFA-Head32-emb256-3s-LRS10-Epoch20-LM3-LM
# exp_dir=exp/LargeModel/WavLM-Large-FullFineTuning-MHFA-Head32-emb256-3s-LRS10-Epoch15-Finetuning
# exp_dir=exp/CA-MHFA/WavLM-BasePlus-FullFineTuning-G-MHFA_Conv2D-emb256-3s-LRS10-Epoch40-Head64-L4-LM
# gpus="[0,1,2,3]"
gpus="[0,1,2,3]"

# gpus="[0]"

num_avg=3
checkpoint=

trials="vox1_O_cleaned.kaldi"
score_norm_method="asnorm"  # asnorm/snorm
top_n=300

# setup for large margin fine-tuning
lm_config=conf/wavlm_base_MHFA_LR_lm.yaml

. tools/parse_options.sh || exit 1

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  echo "Prepare datasets ..."
  ./local/prepare_data.sh --stage 4 --stop_stage 4 --data ${data}
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  echo "Covert train and test data to ${data_type}..."
  for dset in vox1_O; do
    if [ $data_type == "shard" ]; then
      python tools/make_shard_list.py --num_utts_per_shard 1000 \
          --num_threads 4 \
          --prefix shards \
          --shuffle \
          ${data}/$dset/wav.scp ${data}/$dset/utt2spk \
          ${data}/$dset/shards ${data}/$dset/shard.list
    else
      python tools/make_raw_list.py ${data}/$dset/wav.scp \
          ${data}/$dset/utt2spk ${data}/$dset/raw.list
    fi
  done
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  echo "******This script is used to fast evaluate model performance on Vox1-O*****"
  for i in $(seq 1 -1 0); do
    model_path=$exp_dir/models/model_$i.pt

    echo $model_path
    echo "Extract embeddings ..."
    local/extract_vox_tmp.sh \
      --exp_dir $exp_dir --model_path $model_path \
      --nj 4 --gpus $gpus --data_type $data_type --data $data

    echo "Score ..."
    local/score_voxO.sh \
      --stage 1 --stop-stage 2 \
      --data ${data} \
      --exp_dir $exp_dir \
      --trials $trials
    done
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
  for i in $(seq 2 -1 1); do
    model_path=$exp_dir/models/model_$i.pt

    echo $model_path
    echo "Extract embeddings ..."
    local/extract_vox_tmp_short.sh \
      --exp_dir $exp_dir --model_path $model_path \
      --nj 4 --gpus $gpus --data_type $data_type --data $data

    echo "Score ..."
    local/score_voxO_short.sh \
      --stage 1 --stop-stage 2 \
      --data ${data} \
      --exp_dir $exp_dir \
      --trials $trials
    done
fi
