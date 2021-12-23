#!/bin/bash


#for alg in "UNIF" "GIGA-REC" "GIGA-REAL" "GIGA-REAL-EXACT" "GIGA-OPT" "GIGA-OPT-EXACT" "SVI-EXACT" "SVI"
for alg in "UNIF" "GIGA-REAL" "GIGA-OPT" "SVI" "ANC"
do
    for ID in {1..3}
    do

        python3 main.py --alg $alg --trial $ID run
    done
done
