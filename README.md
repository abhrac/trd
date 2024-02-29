# Transitivity Recovering Decompositions: Interpretable and Robust Fine-Grained Relationships
Official implementation of "[Transitivity Recovering Decompositions: Interpretable and Robust Fine-Grained Relationships](https://openreview.net/forum?id=wUNPmdE273)", NeurIPS 2023.

## Environment Setup

This project is implemented using PyTorch. A pip environment with all related dependencies can be created as follows:
1. Clone the project repository:
```shell
git clone https://github.com/abhrac/trd.git
cd trd
```
2. Install dependencies and activate conda environment:
```shell
conda env create -f environment.yml
conda activate trd
```
3. Run:
```shell
python3 src/main.py --dataset='FGVCAircraft' --seed=0 --model-type=multiview_hausdorff --train-backbone --crop-mode=random --local-weight=1e-4 --train-bsize=8 --gpu=1 --recovery-epoch=1
```
The `run_expt.sh` file contains sample training commands.

## Citation
```
@inproceedings{
  chaudhuri2023TRD,
  title={Transitivity Recovering Decompositions: Interpretable and Robust Fine-Grained Relationships},
  author={Abhra Chaudhuri and Massimiliano Mancini and Zeynep Akata and Anjan Dutta},
  booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
  year={2023},
  url={https://openreview.net/forum?id=wUNPmdE273}
}
```
