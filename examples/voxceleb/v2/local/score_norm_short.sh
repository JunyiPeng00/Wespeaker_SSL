#!/bin/bash

# Copyright (c) 2022 Chengdong Liang (liangchengdong@mail.nwpu.edu.cn)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

score_norm_method="asnorm"  # asnorm/snorm
cohort_set=vox2_dev
cohort_set_e=vox2_dev_shortU
top_n=100
exp_dir=
trials="vox1_O_cleaned.kaldi vox1_E_cleaned.kaldi vox1_H_cleaned.kaldi"
data=data

stage=-1
stop_stage=-1

. tools/parse_options.sh
. path.sh


if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
  echo "compute mean xvector"
  python tools/vector_mean.py \
    --spk2utt ${data}/${cohort_set}/spk2utt \
    --xvector_scp $exp_dir/embeddings/${cohort_set_e}/xvector.scp \
    --spk_xvector_ark $exp_dir/embeddings/${cohort_set_e}/spk_xvector.ark
fi

output_name=${cohort_set_e}_${score_norm_method}
[ "${score_norm_method}" == "asnorm" ] && output_name=${output_name}${top_n}
if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
  echo "compute norm score"
  for x in $trials; do
    python wespeaker/bin/score_norm_short.py \
      --score_norm_method $score_norm_method \
      --top_n $top_n \
      --trial_score_file $exp_dir/scores_tmp/${x}_tmp_short.score \
      --score_norm_file $exp_dir/scores_tmp/${output_name}_${x}.score \
      --cohort_emb_scp ${exp_dir}/embeddings/${cohort_set_e}/spk_xvector.scp \
      --eval_emb_scp ${exp_dir}/embeddings/vox1_shortU/xvector.scp \
      --mean_vec_path ${exp_dir}/embeddings/vox2_dev_shortU/mean_vec.npy
  done
fi

if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
  echo "compute metrics"
  for x in ${trials}; do
    scores_dir=${exp_dir}/scores_tmp
    python wespeaker/bin/compute_metrics.py \
      --p_target 0.01 \
      --c_fa 1 \
      --c_miss 1 \
      ${scores_dir}/${output_name}_${x}.score \
      2>&1 | tee -a ${scores_dir}/vox1_${score_norm_method}${top_n}_result

    python wespeaker/bin/compute_det.py \
      ${scores_dir}/${output_name}_${x}.score
  done
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
  echo "compute metrics (EER/minDCF) ..."
  scores_dir1=${exp_dir}/scores/
  scores_dir2=${exp_dir}/scores_tmp

  for x in $trials; do
    python wespeaker/bin/combine_score.py \
          --input1 ${scores_dir1}/vox2_dev_asnorm300_${x}.score\
          --input2 ${scores_dir2}/${output_name}_${x}.score\
          --output ${scores_dir2}/${output_name}_${x}_mix.score\

    python wespeaker/bin/compute_metrics.py \
        --p_target 0.01 \
        --c_fa 1 \
        --c_miss 1 \
        ${scores_dir2}/${output_name}_${x}_mix.score \
        2>&1 | tee -a ${scores_dir2}/vox1_${score_norm_method}${top_n}_result_mix

    # echo "compute DET curve ..."
    # python wespeaker/bin/compute_det.py \
    #     ${scores_dir2}/${x}_mix.score
  done
fi