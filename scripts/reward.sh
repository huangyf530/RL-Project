# !/bin/bash

srun -G 1 --nodelist=101server -p rtx2080 --mem 15G --pty python show_reward.py $1