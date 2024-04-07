# particle_track_reconstruction
Transformer-inspired model for particle track reconstruction in high energy physics

Running the code on Snellius:
```
module purge
module load 2022
module load PyTorch/1.12.0-foss-2022a-CUDA-11.7.0
virtualenv .venv
source .venv/bin/activate
pip install pandas
pip install toml
pip install wandb
pip install git+https://github.com/LAL/trackml-library.git

cd <path_to_src>
python train.py <path_to_toml_file>
```
