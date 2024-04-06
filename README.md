# particle_track_reconstruction
Transformer-inspired model for particle track reconstruction in high energy physics

Running the code on Snellius:
```
module purge
module load 2022
module load PyTorch/1.12.0-foss-2022a-CUDA-11.7.0
virtualenv .venv
source .venv/bin/activate
pip3 install pandas
pip3 install toml
pip3 install wandb
pip3 install --user git+https://github.com/LAL/trackml-library.git
(for the last install to work, one needs to go to the `pyvenv.cfg` file in the Virtual environment folder and set the `include-system-site-packages` to `true`)

cd <path_to_src>
python train.py <path_to_toml_file>
```
