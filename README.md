# WeSpeaker_SSL_BASED_SPEAKER_VERIFICATION

For more details, you can find in `examples/voxceleb/v2`.

### Install for development & deployment
* Clone this repo
``` sh
git clone https://github.com/wenet-e2e/wespeaker.git
```

* Create conda env: pytorch version >= 1.10.0 is required !!!
``` sh
conda create -n wespeaker python=3.9
conda activate wespeaker
conda install pytorch=1.12.1 torchaudio=0.12.1 cudatoolkit=11.3 -c pytorch -c conda-forge
pip install -r requirements.txt
```