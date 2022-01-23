#!/bin/bash

for alg in "LAP" "UNIF" "GIGA" "IHT" "QNC"
#for alg in "QNC"
do
    for ID in 16
    do
    	for M in 500 1000 2000
    	do
			python3 main.py --data_num 10000 --n_bases_per_scale 50 --coreset_size $M --alg $alg --trial $ID run
		done
	done
done

