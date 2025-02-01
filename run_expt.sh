#!/usr/bin/env bash
# Example Training
# python3 src/main.py --dataset='FGVCAircraft' --seed=0 --model-type=proxy_graph --train-backbone --crop-mode=random --local-weight=1 --train-bsize=8 --gpu=0 --backbone-type=disjoint_encoder
# python3 src/main.py --dataset='FGVCAircraft' --seed=0 --model-type=multiview_hausdorff --train-backbone --crop-mode=random --local-weight=1e-4 --train-bsize=8 --gpu=1 --recovery-epoch=1
# python3 src/main.py --dataset='FGVCAircraft' --seed=0 --model-type=gnn_agg_hausdorff --train-backbone --crop-mode=random --local-weight=1 --train-bsize=16 --gpu=1
# python3 src/main.py --dataset='FGVCAircraft' --seed=0 --model-type=transformer_agg --train-backbone --crop-mode=random --local-weight=1
# python3 src/main.py --dataset='FGVCAircraft' --seed=0 --pretrained
# python3 src/main.py --dataset='StanfordCars' --seed=0
# python3 src/main.py --dataset='CUB' --seed=0 --pretrained
# python3 src/main.py --dataset='NABirds' --seed=0 --pretrained
# python3 src/main.py --dataset='iNaturalist' --seed=0 --pretrained
# python3 src/main.py --dataset='CottonCultivar' --seed=0 --pretrained
# python3 src/main.py --dataset='SoyCultivar' --seed=0 --pretrained

# Example Evaluation
# python3 src/main.py --dataset='FGVCAircraft' --seed=0 --model-type=proxy_graph --train-backbone --crop-mode=random --local-weight=1 --train-bsize=8 --gpu=0 --backbone-type=disjoint_encoder --pretrained --eval_only
# python3 src/main.py --dataset='CUB' --seed=0 --model-type=proxy_graph --train-backbone --crop-mode=random --local-weight=1 --train-bsize=8 --gpu=1 --backbone-type=disjoint_encoder --pretrained --eval_only
# python3 src/main.py --dataset='StanfordCars' --seed=0 --model-type=proxy_graph --train-backbone --crop-mode=random --local-weight=1 --train-bsize=8 --gpu=1 --backbone-type=disjoint_encoder --pretrained --eval_only
# python3 src/main.py --dataset='CottonCultivar' --seed=0 --model-type=proxy_graph --train-backbone --crop-mode=random --local-weight=1 --train-bsize=8 --gpu=2 --backbone-type=disjoint_encoder --pretrained --eval_only
# python3 src/main.py --dataset='SoyCultivar' --seed=0 --model-type=proxy_graph --train-backbone --crop-mode=random --local-weight=1 --train-bsize=8 --gpu=3 --backbone-type=disjoint_encoder --pretrained --eval_only
