#!/bin/bash

for alg in "LAP" "UNIF" "GIGA" "QNC" "IHT" "SVI"
do
    for ID in 1 2 3 4 5
    do
    	for M in 50 100 500 1000
        do
       		python3 main.py --alg $alg --trial $ID --coreset_size $M run
       	done
    done
done
