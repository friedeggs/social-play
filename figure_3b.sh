#!/bin/sh
# ./figure_3b.sh  13674.74s user 8761.68s system 331% cpu 1:52:48.34 total

# divide by 7 times 10

declare -a arr=(10000 20000 30000 40000 80000 120000)

for seed in `seq 1237 1243`; do
	index=0
	pids=()
	for n_data in "${arr[@]}"; do
		if [ ! -f results-3b-${seed}-${n_data}.txt ]; then
			python figure_3b.py --n_data $n_data --seed $seed & 
			pids[${index}]=$!
			((index++))
		fi
	done

	for pid in ${pids[*]}; do
	    wait $pid
	done
done

