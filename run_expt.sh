#!/usr/bin/env bash
#python3 src/main.py --dataset='CUB' --seed=0 --backbone=vit_small_patch16_384 --method rproxy --pretrained --epochs=100 --batch_size=8
python3 src/main.py --dataset='StanfordCars' --seed=0 --backbone=vit_small_patch16_384 --method graphrel_hausdorff --pretrained --epochs=100 --batch_size=8
#python3 src/main.py --dataset='NABirds' --seed=0 --backbone=vit_small_patch16_384 --method rproxy --pretrained --epochs=100 --batch_size=8
#python3 src/main.py --dataset='StanfordDogs' --seed=0 --backbone=vit_small_patch16_384 --method graphrel_hausdorff --pretrained --epochs=100 --batch_size=8
#python3 src/main.py --dataset='FGVCAircraft' --seed=0 --backbone=vit_small_patch16_384 --method rproxy --pretrained --epochs=100 --batch_size=8
#python3 src/main.py --dataset='FGVCAircraft' --seed=0 --backbone=vit_small_patch16_384 --method graphrel_hausdorff --pretrained --epochs=100 --batch_size=4

