#!/usr/bin/env bash
python3 src/main.py --dataset='FGVCAircraft' --seed=0 --model-type=gnn_agg_hausdorff --train-backbone --crop-mode=random --local-weight=1 --train-bsize=16 --gpu=1
# python3 src/main.py --dataset='FGVCAircraft' --seed=0 --model-type=transformer_agg --train-backbone --crop-mode=random --local-weight=1
# python3 src/main.py --dataset='FGVCAircraft' --seed=0 --pretrained
# python3 src/main.py --dataset='StanfordCars' --seed=0
# python3 src/main.py --dataset='CUB' --seed=0 --pretrained
# python3 src/main.py --dataset='NABirds' --seed=0 --pretrained
# python3 src/main.py --dataset='iNaturalist' --seed=0 --pretrained
# python3 src/main.py --dataset='CottonCultivar' --seed=0 --pretrained
# python3 src/main.py --dataset='SoyCultivar' --seed=0 --pretrained
