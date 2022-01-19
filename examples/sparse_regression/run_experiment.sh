#!/bin/bash

for dnm in "synth_sparsereg"
do
    for alg in "UNIF" "GIGA" "IHT" "QNC" "SVI"
    do
        for ID in 1 2 3
        do
        	for M in 20 50 100
        	do
				python3 main.py --dataset $dnm --coreset_size $M --alg $alg --trial $ID run
			done
		done
    done
done

