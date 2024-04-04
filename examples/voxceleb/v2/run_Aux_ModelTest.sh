#!/bin/bash
#SBATCH -J Improved_SSL_SPK-Encoder
#SBATCH -p gpu-a6000-kishin
#SBATCH --gres=gpu:4
#SBATCH -c 8

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

config=conf/compare/wavlm_base_CorrPoolingDrop.yaml # wavlm_base_MHFA_LR
# config=conf/wavlm_base_MHFA_LR_Pooling_Group4_Conv2D_32_L4_Drop.yaml # wavlm_base_MHFA_LR
# config=conf/large/wavlm_large_MHFA_LR.yaml
config=conf/TMP.yaml # wavlm_base_MHFA_LR
config=conf/V2/wavlm_base_MHFA_LR.yaml 
config=conf/CA_MHFA/wavlm_base_MHFA_LR_Pooling_Group4_Conv2D_32_L4.yaml
config=conf/adapter/wavlm_base_FixedSSL.yaml
config=conf/large/wavlm_large_MHFA_LR_Fixed.yaml

# exp_dir=exp/WavLM-BasePlus-FullFineTuning-MHFA_Context-emb256-3s-LRS10-Epoch40

base_port=1024
max_port=40000
current_time=$(date +%s)
port=$((current_time % (max_port - base_port) + base_port))

exp_dir=exp/TMP_${port}


# gpus="[0,1,2,3]"
gpus="[0,1,2,3]"

# gpus="[0]"

num_avg=3
checkpoint=

trials="vox1_O_cleaned.kaldi vox1_E_cleaned.kaldi vox1_H_cleaned.kaldi"
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
  for dset in vox2_dev vox1; do
    if [ $data_type == "shard" ]; then
      python tools/make_shard_list.py --num_utts_per_shard 1000 \
          --num_threads 32 \
          --prefix shards \
          --shuffle \
          ${data}/$dset/wav.scp ${data}/$dset/utt2spk \
          ${data}/$dset/shards ${data}/$dset/shard.list
    else
      python tools/make_raw_list.py ${data}/$dset/wav.scp \
          ${data}/$dset/utt2spk ${data}/$dset/raw.list
    fi
  done
  # # Convert all musan data to LMDB
  # python tools/make_lmdb.py ${data}/musan/wav.scp ${data}/musan/lmdb
  # # Convert all rirs data to LMDB
  # python tools/make_lmdb.py ${data}/rirs/wav.scp ${data}/rirs/lmdb
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  echo "Start training ..."
  echo "Just for Debuging"
  num_gpus=$(echo $gpus | awk -F ',' '{print NF}')
  torchrun --standalone --nnodes=1 --nproc_per_node=$num_gpus \
    wespeaker/bin/train_V2.py --config $config \
      --exp_dir ${exp_dir} \
      --gpus $gpus \
      --num_avg ${num_avg} \
      --data_type "${data_type}" \
      --train_data ${data}/vox2_dev/${data_type}_tmp.list \
      --train_label ${data}/vox2_dev/utt2spk_tmp \
      --PORT ${port} \
      --model_args:adapter_dim ${adapter_dim} \
      ${checkpoint:+--checkpoint $checkpoint}
fi
