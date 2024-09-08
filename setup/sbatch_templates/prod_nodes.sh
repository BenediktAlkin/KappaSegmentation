#!/bin/bash -l
#SBATCH --nodes={nodes}
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=28
#SBATCH --partition=compute
#SBATCH --gres=gpu:8
#SBATCH --time={time}
#SBATCH --chdir={chdir}
#SBATCH --output={output}

# set the first node name as master address
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
export MASTER_PORT={master_port}
# add all hostnames info for logging
export ALL_HOST_NAMES=$(scontrol show hostnames "$SLURM_JOB_NODELIST")

# activate conda env
conda activate {env_name}

# write python command to log file -> easy check for which run crashed if there is some config issue
echo python main_train.py {cli_args}

# run
srun --kill-on-bad-exit=1 --cpus-per-task 28 python main_train.py {cli_args}