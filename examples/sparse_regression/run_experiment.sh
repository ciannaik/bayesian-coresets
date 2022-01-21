#!/bin/bash

for dnm in "synth_sparsereg_large"
do
    for alg in "UNIF" "GIGA" "IHT"
#    for alg in "QNC"
    do
        for ID in 1 2 3
        do
        	for M in 100 200 500
        	do
				python3 main.py --dataset $dnm --coreset_size $M --alg $alg --trial $ID run
			done
		done
    done
done

