#!/bin/sh
# ./figure_1b.sh  16490.91s user 11474.82s system 133% cpu 5:48:03.32 total
# ./figure_1b.sh  15977.05s user 11796.42s system 292% cpu 2:38:17.76 total


# 6 hours to run

WANDB_PROJECT_NAME=""
WANDB_ENTITY=""

declare -a arr=("reward-model" "oracle" "none")
pids=()

# for seed in `seq 1234 1238`; do
for seed in `seq 1239 1243`; do
	index=0
	for mode in "${arr[@]}"; do
		python figure_1b.py --mode $mode --seed $seed --track --wandb-project-name $WANDB_PROJECT_NAME --wandb-entity $WANDB_ENTITY & 
		pids[${index}]=$!
		((index++))
	done

	for pid in ${pids[*]}; do
	    wait $pid
	done
done

