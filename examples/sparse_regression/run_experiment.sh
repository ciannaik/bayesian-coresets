#!/bin/bash

for dnm in "delays"
do
    for alg in "LAP" "UNIF" "GIGA" "IHT" "QNC"
#    for alg in "LAP"
    do
        for ID in 1
        do
        	for M in 100 500 1000
        	do
				python3 main.py --dataset $dnm --coreset_size $M --alg $alg --trial $ID run
			done
		done
    done
done

