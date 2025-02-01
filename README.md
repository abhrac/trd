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

## Evaluation
To evaluate on a dataset using pretrained weights, first download the model for the corresponding dataset from
[here](https://drive.google.com/drive/folders/1L79fXc8MnvnLA1tOzOlURItyg2QT9hdu?usp=sharing)
and place it under the folder `./checkpoint/$DataSetName/`,
where `./checkpoint` is under the project root, but could optionally be elsewhere too
(see `src/options.py`). Then, run the following command:
```shell
python3 src/main.py --data_root='RootDirForAllDatasets' --dataset='DatasetName' --pretrained --eval_only
```
The `run_expt.sh` file contains sample training commands.

## Disclaimer
The pretrained weights provided can be used to reproduce the results in the paper. However, since the training of the models were done using pretrained weights from prior works ([10] and [94] in the main paper), and it consisted of several phases with slightly different hyperparameters for each phase, not all of which were always kept track of, running the training scripts from scratch with the default hyperparameters provided here is unlikely to produce expected results. The purpose of the training scripts provided here is to illustrate how the end-to-end pipelines were implemented. The purpose of the pretrained weights provided here is to reproduce the actual results.

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
