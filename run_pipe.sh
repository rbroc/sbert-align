# !/bin/bash

#python3 ./scripts/get_surrogate_ids.py --sampled-ids 50

for lag in 1 2 3
do
    for model in all-MiniLM-L6-v2 all-mpnet-base-v2
    do
        for ptype in surrogate true
        do
            python3 align.py --lag $lag --model $model --pair-type $ptype
        done
    done
done

git add outputs/*
git commit -am 'add output files'
git push 
