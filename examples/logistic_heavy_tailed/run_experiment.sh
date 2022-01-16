#!/bin/bash

for dnm in "synth_lr_cauchy"
do
    for alg in "UNIF" "GIGA" "ANC"
    do
        for ID in 15
        do
        	for M in 10 50 100
        	do
				python3 main.py --model lr --dataset $dnm --coreset_size $M --alg $alg --trial $ID run
			done
		done
    done
done

