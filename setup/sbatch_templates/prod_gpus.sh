#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks-per-node={gpus}
#SBATCH --cpus-per-task=28
#SBATCH --partition=compute
#SBATCH --gres=gpu:{gpus}
#SBATCH --time={time}
#SBATCH --chdir={chdir}
#SBATCH --output={output}

export MASTER_ADDR=localhost
export MASTER_PORT={master_port}

# activate conda env
conda activate {env_name}

# write python command to log file -> easy check for which run crashed if there is some config issue
echo python main_train.py {cli_args}

# run
srun --kill-on-bad-exit=1 --cpus-per-task 28 python main_train.py {cli_args}