#!/bin/bash
#SBATCH -J Improved_SSL_SPK-Encoder-Baseline-CM
#SBATCH -p gpu-2080ti-kishin
#SBATCH --gres=gpu:4
#SBATCH -c 4
WORK_DIR=/home/jpeng/ntt/work/SPKID/examples/cnceleb/v2

source /home/jpeng/anaconda3/bin/activate /home/jpeng/anaconda3/envs/wespeaker

# which python
cd $WORK_DIR


. ./path.sh || exit 1

base_port=1024
max_port=40000
current_time=$(date +%s)
port=$((current_time % (max_port - base_port) + base_port))

stage=4
stop_stage=6

data=data
data_type="shard"  # shard/raw

# config=conf/wavlm_base_MHFA_LR3_CNNFixed.yaml
# exp_dir=exp/WavLM_BASE_PLUS-MHFA-Epoch30-CNNFixed

# config=conf/wavlm_base_MHFA_LR3.yaml
# exp_dir=exp/WavLM_BASE_PLUS-MHFA-Epoch20
# exp_dir=exp/TMP/TMP_${port}

config=conf/wavlm_base_CA_MHFA_LR3_CNNFixed.yaml
# exp_dir=exp/WavLM_BASE_PLUS-CA-MHFA-Epoch30-CNNFixed

# exp_dir=exp/WavLM_BASE_PLUS-CA-MHFA-Epoch30-CNNFixed-LM
exp_dir=exp/CN1/WavLM_BASE_PLUS-MHFA-Epoch30-CNNFixed-Vox2-Adapter-LM
gpus="[0,1,2,3]"
num_avg=1
checkpoint=

trials="CNC-Eval-Concat.lst CNC-Eval-Avg.lst"
score_norm_method="asnorm"  # asnorm/snorm
top_n=300

# setup for large margin fine-tuning
lm_config=conf/wavlm_base_CA_MHFA_LR3_CNNFixed_lm.yaml

. tools/parse_options.sh || exit 1

# if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
#   echo "Preparing datasets ..."
#   ./local/prepare_data.sh --stage 2 --stop_stage 4 --data ${data}
# fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  echo "Covert train and test data to ${data_type}..."
  for dset in cnceleb_train eval; do
    if [ $data_type == "shard" ]; then
      python tools/make_shard_list.py --num_utts_per_shard 1000 \
          --num_threads 16 \
          --prefix shards \
          --shuffle \
          ${data}/$dset/wav.scp ${data}/$dset/utt2spk \
          ${data}/$dset/shards ${data}/$dset/shard.list
    else
      python tools/make_raw_list.py ${data}/$dset/wav.scp \
          ${data}/$dset/utt2spk ${data}/$dset/raw.list
    fi
  done
  # Convert all musan data to LMDB
  python tools/make_lmdb.py ${data}/musan/wav.scp ${data}/musan/lmdb
  # Convert all rirs data to LMDB
  python tools/make_lmdb.py ${data}/rirs/wav.scp ${data}/rirs/lmdb
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  echo "Start training ..."
  num_gpus=$(echo $gpus | awk -F ',' '{print NF}')
  torchrun --standalone --nnodes=1 --nproc_per_node=$num_gpus \
    wespeaker/bin/train_V2.py --config $config \
      --exp_dir ${exp_dir} \
      --gpus $gpus \
      --num_avg ${num_avg} \
      --data_type "${data_type}" \
      --train_data ${data}/cnceleb_train/${data_type}.list \
      --train_label ${data}/cnceleb_train/utt2spk \
      --reverb_data ${data}/rirs/lmdb \
      --noise_data ${data}/musan/lmdb \
      --master_port ${port} \
      ${checkpoint:+--checkpoint $checkpoint}
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
  echo "Do model average ..."
  avg_model=$exp_dir/models/avg_model.pt
  python wespeaker/bin/average_model.py \
    --dst_model $avg_model \
    --src_path $exp_dir/models \
    --num ${num_avg}

  model_path=$avg_model

  echo "Extract embeddings ..."
  local/extract_cnc_cn1.sh \
    --exp_dir $exp_dir --model_path $model_path \
    --nj 4 --gpus $gpus --data_type $data_type --data ${data}
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
  echo "Score ..."
  local/score_cn1.sh \
    --stage 1 --stop-stage 2 \
    --exp_dir $exp_dir \
    --data ${data} \
    --trials "$trials"
fi

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
  echo "Score norm ..."
  local/score_norm_cn1.sh \
    --stage 1 --stop-stage 3 \
    --score_norm_method $score_norm_method \
    --cohort_set cnceleb_train \
    --top_n $top_n \
    --exp_dir $exp_dir \
    --data ${data} \
    --trials "$trials"
fi
