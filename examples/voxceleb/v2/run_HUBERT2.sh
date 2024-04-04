#!/bin/bash
#SBATCH -J Improved_SSL_SPK-Encoder-V2
#SBATCH -p gpu-a6000-kishin
#SBATCH --gres=gpu:4
#SBATCH -c 4

WORK_DIR=/home/jpeng/ntt/work/SPKID/examples/voxceleb/v2

source /home/jpeng/anaconda3/bin/activate /home/jpeng/anaconda3/envs/wespeaker

# which python
cd $WORK_DIR

. ./path.sh || exit 1


stage=3 #3
stop_stage=6 #6

data=data
data_type="shard"  # shard/raw

config=conf/V2/wavlm_base_MHFA_LR.yaml 


exp_dir=exp/WavLM-BasePlus-FullFineTuning-MHFA-emb256-3s-LRS10-Epoch40


base_port=1024
max_port=40000
current_time=$(date +%s)
port=$((current_time % (max_port - base_port) + base_port))

gpus="[0,1,2,3]"

# gpus="[0]"

num_avg=3
checkpoint=

trials="vox1_O_cleaned.kaldi vox1_E_cleaned.kaldi vox1_H_cleaned.kaldi"
score_norm_method="asnorm"  # asnorm/snorm
top_n=300

# setup for large margin fine-tuning
lm_config=conf/wavlm_base_MHFA_LR_lm2.yaml

. tools/parse_options.sh || exit 1

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  echo "Prepare datasets ..."
  ./local/prepare_data.sh --stage 4 --stop_stage 4 --data ${data}
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  echo "Covert train and test data to ${data_type}..."
  # for dset in vox2_dev vox1; do
  #   if [ $data_type == "shard" ]; then
  #     python tools/make_shard_list.py --num_utts_per_shard 1000 \
  #         --num_threads 32 \
  #         --prefix shards \
  #         --shuffle \
  #         ${data}/$dset/wav.scp ${data}/$dset/utt2spk \
  #         ${data}/$dset/shards ${data}/$dset/shard.list
  #   else
  #     python tools/make_raw_list.py ${data}/$dset/wav.scp \
  #         ${data}/$dset/utt2spk ${data}/$dset/raw.list
  #   fi
  # done
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
      --train_data ${data}/vox2_dev/${data_type}.list \
      --train_label ${data}/vox2_dev/utt2spk \
      --reverb_data ${data}/rirs/lmdb \
      --noise_data ${data}/musan/lmdb \
      --PORT ${port} \
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
  local/extract_vox.sh \
    --exp_dir $exp_dir --model_path $model_path \
    --nj 16 --gpus $gpus --data_type $data_type --data ${data}
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
  echo "Score ..."
  local/score.sh \
    --stage 1 --stop-stage 2 \
    --data ${data} \
    --exp_dir $exp_dir \
    --trials "$trials"
fi

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
  echo "Score norm ..."
  local/score_norm.sh \
    --stage 1 --stop-stage 3 \
    --score_norm_method $score_norm_method \
    --cohort_set vox2_dev \
    --top_n $top_n \
    --data ${data} \
    --exp_dir $exp_dir \
    --trials "$trials"
fi

if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
  echo "Export the best model ..."
  # python wespeaker/bin/export_jit.py \
  #   --config $exp_dir/config.yaml \
  #   --checkpoint $exp_dir/models/avg_model.pt \
  #   --output_file $exp_dir/models/final.zip
fi

if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ]; then
  echo "Large margin fine-tuning ..."
  lm_exp_dir=${exp_dir}-LM2
  mkdir -p ${lm_exp_dir}/models
  # Use the pre-trained average model to initialize the LM training
  cp ${exp_dir}/models/avg_model.pt ${lm_exp_dir}/models/model_0.pt
  bash run_HUBERT2.sh --stage 3 --stop_stage 7 \
      --data ${data} \
      --data_type ${data_type} \
      --config ${lm_config} \
      --exp_dir ${lm_exp_dir} \
      --gpus $gpus \
      --num_avg 1 \
      --checkpoint ${lm_exp_dir}/models/model_0.pt \
      --trials "$trials" \
      --score_norm_method ${score_norm_method} \
      --top_n ${top_n}
fi