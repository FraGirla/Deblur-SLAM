#!/bin/bash
#SBATCH --time=72:00:00
#SBATCH --mem-per-cpu=80000
#SBATCH --tmp=8000                        # per node!!
#SBATCH --job-name=Deblur-SLAM
#SBATCH --account=ls_polle
#SBATCH --output=slurm_logs/Deblur-SLAM-%j.out
#SBATCH --gpus=1
#SBATCH --gres=gpumem:20g
#module load gcc/6.3.0 python_gpu/3.7.4 cuda/11.6.2 cudnn/8.4.0.27 cmake/3.20.3
#module load gcc/11.4.0 python_gpu/3.11.6 cuda/12.1.1 cudnn/8.9.2.26
#module load gcc/8.2.0 python_gpu/3.11.2 cuda/11.7.0 cudnn/8.4.0.27

module load stack/2024-06
module load python_cuda/3.11.6 #eth_proxy
#nvidia-smi
#nvcc --version

cd /cluster/home/fgirlanda/Deblur-SLAM/

source ../MonoGS/venv311/bin/activate
lscpu | grep "Model name:" | sed -r 's/Model name:\s{1,}//g'

#python slam.py --config configs/mono/replica_ext/office0_sp.yaml
#python slam.py --config configs/mono/replica_ext/office2_sp.yaml
#python run.py configs/ReplicaBlurry/office0.yaml

#python run.py configs/TUM_RGB/freiburg1_desk.yaml
#python run.py configs/TUM_RGB/freiburg2_xyz.yaml
#python run.py configs/TUM_RGB/freiburg3_office.yaml
#python run.py configs/TUM_RGB/freiburg1_desk2.yaml
#python run.py configs/TUM_RGB/freiburg1_room.yaml

#python run.py configs/Scannet/scene0000.yaml
#python run.py configs/Scannet/scene0059.yaml
#python run.py configs/Scannet/scene0106.yaml
#python run.py configs/Scannet/scene0169.yaml
#python run.py configs/Scannet/scene0181.yaml
#python run.py configs/Scannet/scene0207.yaml

python run.py configs/ETH3D/sofa_2_ext2.yaml
python run.py configs/ETH3D/repetitive_ext2.yaml
python run.py configs/ETH3D/plant_scene_2_ext2.yaml
python run.py configs/ETH3D/plant_scene_3_ext2.yaml

#python run.py configs/ETH3D/sofa_1_ext2.yaml
#python run.py configs/ETH3D/plant_scene_1_ext2.yaml
#python run.py configs/ETH3D/table_3_ext2.yaml
#python run.py configs/ETH3D/sfm_bench_ext2.yaml
#python run.py configs/ETH3D/sfm_garden_ext2.yaml
#python run.py configs/ETH3D/sfm_lab_room_1_ext2.yaml
#python run.py configs/ETH3D/sfm_lab_room_2_ext2.yaml
#python run.py configs/ETH3D/sfm_house_loop_ext2.yaml