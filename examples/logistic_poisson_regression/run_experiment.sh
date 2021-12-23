#!/bin/bash

for dnm in "ds1"
do
    for alg in "UNIF" "GIGA-OPT" "GIGA-REAL" "GIGA-REC-MCMC" "ANC" "SVI"
#    for alg in "ANC"
    do
        for ID in {1..3}
        do
		python3 main.py --model lr --dataset $dnm --alg $alg --trial $ID run
	done
    done
done

