#!/bin/bash

for alg in "LAP" "UNIF" "GIGA" "QNC" "SVI"
do
    for ID in 1 2 3
    do
    	for M in 10 20 50 100
        do
       		python3 main.py --alg $alg --trial $ID --coreset_size $M --data_num 1000 --data_dim 10 run
       	done
    done
done
