#!/bin/bash

## Full inference run for each trial (to establish a noise floor for evaluation metrics)
## Also do laplace
## Both of these cache results and re-use across coreset sizes later on during plotting
#for alg in "FULL" "LAP"
#do
#    for ID in {1..10}
#    do
#        python3 main.py --samples_inference 10000 --alg $alg --trial $ID --coreset_size 10 run
#    done
#done

## Run all the other algorithms
#for alg in "QNC" "UNIF" "GIGA" "IHT" "FULL" "LAP"
##for alg in "QNC"
#do
#    for ID in {1..10}
#    do
#    	for M in 100 500 1000 5000 10000
##    	for M in 10 100 1000
#        do
#       		python3 main.py --samples_inference 10000 --alg $alg --trial $ID --coreset_size $M --proj_dim 500 run
#       	done
#    done
#done


# Run all the other algorithms
for alg in "QNC" "UNIF" "GIGA" "IHT"
#
do
    for ID in {1..5}
    do
    	for M in 100 500 1000 5000 10000
#    	for M in 10 100 1000
        do
       		python3 main.py --samples_inference 1000 --alg $alg --trial $ID --coreset_size $M --proj_dim 500 run
       	done
    done
done


## Run naive mcmc algorithms
#for alg in "NAIVE"
##for alg in "QNC"
#do
#    for ID in {1..2}
#    do
#    	for M in 100 500 1000 5000 10000
##    	for M in 10 100 1000
#        do
#       		python3 main.py --samples_inference 10000 --alg $alg --trial $ID --coreset_size $M --proj_dim 500 run
#       	done
#    done
#done
## Run all the other algorithms
#for alg in "SVI"
#do
#    for ID in 1
#    do
#    	for M in 10 50 100 500 1000
#        do
#       		python3 main.py --samples_inference 10000 --alg $alg --trial $ID --coreset_size $M --proj_dim 2000 run
#       	done
#    done
#done

