## Results

* Setup: wav, num_frms300, epoch30, ArcMargin_InterTopK, aug_prob0.6
* Scoring: cosine (sub mean of vox2_dev)
* Metric: EER(%)

| Model | Params | AS-Norm(300) | vox1-O-clean | vox1-E-clean | vox1-H-clean |
|:------|:------:|:------------:|:------------:|:------------:|:------------:|
| XVEC-TSTP-emb512 | 4.61M | × | 1.962 | 1.918 | 3.389 |
|                  |       | √ | 1.835 | 1.822 | 3.110 |
| ECAPA_TDNN_GLOB_c512-ASTP-emb192 | 6.19M | × | 1.149 | 1.248 | 2.313 |
|                                  |       | √ | 1.026 | 1.154 | 2.089 |
| ResNet34-TSTP-emb256 | 6.63M | × | 0.941 | 1.114 | 2.026 |
|                      |       | √ | 0.899 | 1.064 | 1.856 |

* SSL-Based Speaker Verification performance. The weights you can download using this [link](https://drive.google.com/file/d/1F75chja9KAAhfws00kN-sreUv0eeFBri/view?usp=sharing) 
* Training Time: 21 hrs with 4*A6000



| Model | Params | LM | AS-Norm | vox1-O-clean | vox1-E-clean | vox1-H-clean |
|:------|:------:|:--:|:-------:|:------------:|:------------:|:------------:|
| WavLM_BASE_PLUS-MHFA_Head64-emb256  | 94M + 2M | × | × | 0.766 | 0.790 | 1.583 |
|                                   |       | × | √ | 0.744 | 0.825 | 1.672 |