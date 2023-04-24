# PGChat

This is the source code for CIKM 2022 Paper: [Improving Personality Consistency in Conversation by Persona Extending](https://dl.acm.org/doi/abs/10.1145/3511808.3557359)

## Requirements

1. Create a new conda environment.

```shell
conda env create -f environment.yaml
conda activate pgchat
```

2. Download spacy model.

```shell
python -m spacy download en
```

## Usage

### Download Model

The model could be downloaded from both [Google Drive](https://drive.google.com/file/d/1g1J9F6AKkFCdpTKZMaq-vG2gACoFWlMB/view?usp=sharing) and [BaiduNetdisk](https://pan.baidu.com/s/1LB1xK_EAgFaOucKHB5YFUQ?pwd=x6rj). Make sure that the checkpoint is put in `checkpoints/`.

### Train

```shell
python main.py
```

### Inference

Open the `config.py` and set:
```python
'inde_result_predict': True,
```
then:
```shell
python main.py
```
