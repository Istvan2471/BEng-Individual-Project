#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL # required to send email notifcations
#SBATCH --mail-user=id419 # required to send email notifcations - please replace <your_username> with your college login name or email address
export PATH=/vol/bitbucket/id419/projenv/bin/:$PATH
source activate
source /vol/cuda/12.0-cudnn8.0.5.39/setup.sh
xvfb-run -a -s "-screen 0 1400x900x24" python /vol/bitbucket/id419/projenv/individual-project/main.py --symbolic-env
TERM=vt100 # or TERM=xterm
/usr/bin/nvidia-smi
uptime