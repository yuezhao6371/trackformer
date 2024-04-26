# particle_track_reconstruction
Transformer-inspired model for particle track reconstruction in high energy physics

Setting up an environment on Snellius:
```
module purge
module load 2022
module load Python/3.10.4-GCCcore-11.3.0
virtualenv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch
pip install toml
pip install wandb
pip install git+https://github.com/LAL/trackml-library.git
```
Using the virtual environment:
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
Example bash script:
```
#!/bin/bash
#SBATCH --job-name=test
#SBATCH --output=test_%j.out
#SBATCH --error=test_%j.err
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=<your_email@address.com>

module purge
module load 2022
module load PyTorch/1.12.0-foss-2022a-CUDA-11.7.0

cd <path_to_root>
source .venv/bin/activate

python ./src/train.py ./configs/example.toml
```
