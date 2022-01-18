#!/bin/bash

for dnm in "synth_poiss_large"
do
    for alg in "UNIF" "GIGA-OPT" "GIGA-REAL" "ANC"
#    for alg in "ANC"
    do
        for ID in {1..3}
        do
		python3 main.py --model poiss --dataset $dnm --alg $alg --trial $ID run
	done
    done
done

