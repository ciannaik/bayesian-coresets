#!/bin/bash

# Full inference run for each trial (to establish a noise floor for evaluation metrics)
# Also do laplace 
# Both of these cache results and re-use across coreset sizes later on during plotting
for alg in "FULL" "LAP"
do
    for ID in {1..10}
    do
        python3 main.py --samples_inference 1000 --alg FULL --trial $ID run
    done
done

# Run all the other algorithms
for alg in "GIGA" "IHT"
do
    for ID in {1..10}
    do
    	for M in 10 50 100 500 1000 5000
        do
       		python3 main.py --samples_inference 1000 --alg $alg --trial $ID --coreset_size $M --proj_dim 20 run
       	done
    done
done


#for alg in "LAP" "UNIF" "GIGA" "QNC" "IHT"
