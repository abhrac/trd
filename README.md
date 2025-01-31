# Transitivity Recovering Decompositions: Interpretable and Robust Fine-Grained Relationships
Official implementation of "[Transitivity Recovering Decompositions: Interpretable and Robust Fine-Grained Relationships](https://openreview.net/forum?id=wUNPmdE273)", NeurIPS 2023.

## Environment Setup

This project is implemented using PyTorch and PyTorch Geometric as the backbone. A conda environment with all related dependencies can be created as follows:
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
python3 src/main.py --dataset='DatasetName' --seed=0 --model-type=proxy_graph --train-backbone --crop-mode=random --local-weight=1e-4 --train-bsize=8 --gpu=1 --recovery-epoch=1
```
The `run_expt.sh` file contains sample training commands.

## Evaluation
To evaluate on a dataset using pretrained weights, first download the model for the corresponding dataset from
[here](https://drive.google.com/drive/folders/1L79fXc8MnvnLA1tOzOlURItyg2QT9hdu?usp=sharing)
and place it under the folder `./checkpoint/$DataSetName/`,
where `./checkpoint` is under the project root, but could optionally be elsewhere too
(see `src/options.py`). Then, run the following command:
```shell
python3 src/main.py --data_root='RootDirForAllDatasets' --dataset='DatasetName' --pretrained --eval_only
```

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
