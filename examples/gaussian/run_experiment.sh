#!/bin/bash

for alg in "LAP" "UNIF" "GIGA" "QNC" "IHT" "SVI"
do
    for ID in 1 2 3
    do
    	for M in 50 100 200
        do
       		python3 main.py --alg $alg --trial $ID --coreset_size $M --data_num 10000 --data_dim 50 run
       	done
    done
done
