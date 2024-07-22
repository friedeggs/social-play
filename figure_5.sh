#!/bin/sh
# ./figure_5.sh  107985.19s user 82933.23s system 420% cpu 12:36:23.90 total
# ./figure_5.sh  8601.14s user 7497.83s system 498% cpu 53:46.57 total
# ./figure_5.sh  227.97s user 48.62s system 487% cpu 56.795 total


# 6 hours to run

WANDB_PROJECT_NAME=""
WANDB_ENTITY=""

# declare -a arr=("oracle")
# pids=()

# # for seed in `seq 1234 1238`; do
# for seed in `seq 1239 1243`; do
# 	for mode in "${arr[@]}"; do
# 		index=0
# 		for env_id in `seq 1 5`; do
# 			python figure_5.py --env_id $env_id --mode $mode --seed $seed --track --wandb-project-name $WANDB_PROJECT_NAME --wandb-entity $WANDB_ENTITY & 
# 			pids[${index}]=$!
# 			((index++))
# 		done

# 		for pid in ${pids[*]}; do
# 		    wait $pid
# 		done
# 	done
# done

# declare -a arr=("reward-model")
# pids=()

# # for seed in `seq 1234 1238`; do
# for seed in `seq 1239 1243`; do
# 	for mode in "${arr[@]}"; do
# 		index=0
# 		for env_id in `seq 1 5`; do
# 			python replicate_figure_5.py --env_id $env_id --mode $mode --seed $seed --track --wandb-project-name $WANDB_PROJECT_NAME --wandb-entity $WANDB_ENTITY & 
# 			pids[${index}]=$!
# 			((index++))
# 		done

# 		for pid in ${pids[*]}; do
# 		    wait $pid
# 		done
# 	done
# done


# declare -a arr=("frozen")
# pids=()

# # for seed in `seq 1234 1238`; do
# index=0
# for seed in `seq 1239 1243`; do
# 	for mode in "${arr[@]}"; do
# 		# for env_id in `seq 1 5`; do
# 		python figure_5.py --env_id 1 --mode $mode --seed $seed --track --wandb-project-name $WANDB_PROJECT_NAME --wandb-entity $WANDB_ENTITY & 
# 		pids[${index}]=$!
# 		((index++))
# 		# done
# 	done
# done

# for pid in ${pids[*]}; do
#     wait $pid
# done

declare -a arr=("frozen")
pids=()

for seed in `seq 1239 1243`; do
	for mode in "${arr[@]}"; do
		index=0
		for env_id in `seq 1 5`; do
			python figure_5_frozen.py --env_id $env_id --mode $mode --seed $seed --track --wandb-project-name $WANDB_PROJECT_NAME --wandb-entity $WANDB_ENTITY & 
			pids[${index}]=$!
			((index++))
		done

		for pid in ${pids[*]}; do
			wait $pid
		done
	done
done

