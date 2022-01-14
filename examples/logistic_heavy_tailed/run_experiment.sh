#!/bin/bash

for dnm in "synth_lr_cauchy"
do
    for alg in "UNIF" "GIGA-OPT" "GIGA-REAL" "ANC"
#    for alg in "ANC"
    do
        for ID in {15}
        do
		python3 main.py --model lr --dataset $dnm --alg $alg --trial $ID run
	done
    done
done

