#!/bin/bash

# TODO first run FULL in a loop for one coreset size
# then run FULL in parallel for all other coreset sizes (same result) 
# then run every other method in parallel

#for alg in "LAP" "UNIF" "GIGA" "QNC" "IHT"
for alg in "UNIF" "FULL"
do
    for ID in 1 2 3
    do
    	for M in 10 50 100 500 1000 5000
        do
       		python3 main.py --samples_inference 1000 --alg $alg --trial $ID --coreset_size $M run
       	done
    done
done
