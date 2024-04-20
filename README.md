# particle_track_reconstruction
Transformer-inspired model for particle track reconstruction in high energy physics

Setting up environment on Snellius:
```
module purge
module load 2022
module load Python/3.10.4-GCCcore-11.3.0
virtualenv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch
pip install flash-attention
pip install toml
pip install wandb
pip install git+https://github.com/LAL/trackml-library.git
```
Running code:
```
module purge
module load 2022
module load Python/3.10.4-GCCcore-11.3.0
source .venv/bin/activate
cd <path_to_src>
```
Running training:
```
python train.py <path_to_toml_file>
```
Running evaluation:
```
python evaluate.py <path_to_toml_file>
```
