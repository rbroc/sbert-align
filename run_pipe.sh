# !/bin/bash

python3 ./scripts/get_surrogate_ids.py --sampled-ids 50
python3 align.py --lag 1 --pair-type true --model all-MiniLM-L6-v2
python3 align.py --lag 1 --pair-type surrogate --model all-MiniLM-L6-v2
python3 align.py --lag 2 --pair-type true --model all-MiniLM-L6-v2

git add outputs/*
git commit -am 'add output files'
git push 