#!/bin/bash

for dnm in "criteo"
do
    for alg in "LAP" "UNIF" "GIGA" "IHT" "QNC" "FULL"
#    for alg in "UNIF"
    do
        for ID in {1..10}
        do
        	for M in 10 50 100 500 1000
        	do
				python3 main.py --model lr --dataset $dnm --coreset_size $M --alg $alg --trial $ID run
			done
		done
    done
done

