#!/bin/bash

for alg in "LAP" "FULL" "UNIF" "IHT" "QNC"
do
    for ID in {1..10}
    do
    	for M in 50 100 500 1000 5000
    	do
			python3 main.py --data_num 100000 --n_bases_per_scale 50 --coreset_size $M --alg $alg --trial $ID --proj_dim 5000 run
		done
	done
done

for alg in "GIGA"
do
    for ID in {1..10}
    do
    	for M in 50 100 500 1000 5000
    	do
			python3 main.py --data_num 100000 --n_bases_per_scale 50 --coreset_size $M --alg $alg --trial $ID --proj_dim 500 run
		done
	done
done

