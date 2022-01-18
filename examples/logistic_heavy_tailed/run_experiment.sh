#!/bin/bash

for dnm in "synth_lr_cauchy_large"
do
    for alg in "LAP" "UNIF" "GIGA" "QNC"
#    for alg in "LAP"
    do
        for ID in 4
        do
        	for M in 100 200 500 1000
        	do
				python3 main.py --model lr --dataset $dnm --coreset_size $M --alg $alg --trial $ID run
			done
		done
    done
done

