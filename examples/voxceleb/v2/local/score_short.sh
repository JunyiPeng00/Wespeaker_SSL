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

exp_dir=
trials="vox1_O_cleaned.kaldi"
data=data

stage=-1
stop_stage=-1

. tools/parse_options.sh
. path.sh

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  echo "apply cosine scoring ..."
  mkdir -p ${exp_dir}/scores
  trials_dir=${data}/vox1/trials
  for x in $trials; do
    echo $x
    python wespeaker/bin/score_tmp_short.py \
      --exp_dir ${exp_dir} \
      --eval_scp_path ${exp_dir}/embeddings/vox1_shortU/xvector.scp \
      --cal_mean True \
      --cal_mean_dir ${exp_dir}/embeddings/vox2_dev_shortU \
      ${trials_dir}/${x}
  done
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  echo "compute metrics (EER/minDCF) ..."
  scores_dir=${exp_dir}/scores_tmp
  for x in $trials; do
    python wespeaker/bin/compute_metrics.py \
        --p_target 0.01 \
        --c_fa 1 \
        --c_miss 1 \
        ${scores_dir}/${x}_tmp_short.score \
        2>&1 | tee -a ${scores_dir}/vox1_cos_result_short

    echo "compute DET curve ..."
    python wespeaker/bin/compute_det.py \
        ${scores_dir}/${x}_tmp_short.score
  done
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  echo "compute metrics (EER/minDCF) ..."
  scores_dir1=${exp_dir}/scores
  scores_dir2=${exp_dir}/scores_tmp

  for x in $trials; do
    python wespeaker/bin/combine_score.py \
          --input1 ${scores_dir1}/${x}.score\
          --input2 ${scores_dir2}/${x}_tmp_short.score\
          --output ${scores_dir2}/${x}_mix.score\

    python wespeaker/bin/compute_metrics.py \
        --p_target 0.01 \
        --c_fa 1 \
        --c_miss 1 \
        ${scores_dir2}/${x}_mix.score \
        2>&1 | tee -a ${scores_dir2}/vox1_cos_result_mix

    echo "compute DET curve ..."
    python wespeaker/bin/compute_det.py \
        ${scores_dir2}/${x}_mix.score
  done
fi
