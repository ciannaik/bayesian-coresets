#!/bin/bash

for alg in "UNIF" "GIGA" "QNC"
do
    for ID in 1 2 3
    do
    	for M in 10 50 100
        do
       		python3 main.py --alg $alg --trial $ID --coreset_size $M --data_num 1000 --data_dim 10 run
       	done
    done
done
