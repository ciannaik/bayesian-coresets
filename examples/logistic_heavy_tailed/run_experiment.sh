#!/bin/bash

#for dnm in "delays"
#do
#    for alg in "LAP" "UNIF" "GIGA" "IHT" "QNC" "FULL"
##    for alg in "UNIF"
#    do
#        for ID in {1..10}
#        do
#        	for M in 10 50 100 500 1000
#        	do
#				python3 main.py --model lr --dataset $dnm --coreset_size $M --alg $alg --trial $ID run
#			done
#		done
#    done
#done


for dnm in "delays_large"
do
#    for alg in "LAP" "UNIF" "GIGA" "IHT" "QNC"
    for alg in "QNC" "UNIF"
    do
#        for ID in {1..10}
        for ID in 1
        do
        	for M in 100 500 1000 5000 10000
        	do
				python3 main.py --model lr --dataset $dnm --coreset_size $M --alg $alg --trial $ID run
			done
		done
    done
done
