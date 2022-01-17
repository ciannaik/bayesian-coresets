#!/bin/bash

for dnm in "synth_lr_cauchy"
do
    for alg in "UNIF" "GIGA" "QNC"
    do
        for ID in 15 3 5 10
        do
        	for M in 10 50 100
        	do
				python3 main.py --model lr --dataset $dnm --coreset_size $M --alg $alg --trial $ID run
			done
		done
    done
done

