#!/bin/bash

#for alg in "LAP" "UNIF" "GIGA" "QNC" "IHT"
for alg in "UNIF"
do
    for ID in 1 2 3 4 5
    do
    	for M in 10 50 100 500 1000 5000
        do
       		python3 main.py --alg $alg --trial $ID --coreset_size $M run
       	done
    done
done
