#!/bin/bash

for alg in "UNIF" "GIGA" "QNC"
do
    for ID in 15 3 5 10
    do
    	for M in 10 50 100
    	do
			python3 main.py --data_num 500 --n_bases_per_scale 50 --coreset_size $M --alg $alg --trial $ID run
		done
	done
done

