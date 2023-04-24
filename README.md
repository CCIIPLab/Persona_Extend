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

## Citation

```
@inproceedings{10.1145/3511808.3557359,
    author = {Liu, Yifan and Wei, Wei and Liu, Jiayi and Mao, Xianling and 
Fang, Rui and Chen, Dangyang},
    title = {Improving Personality Consistency in Conversation by Persona 
Extending},
    year = {2022},
    isbn = {9781450392365},
    publisher = {Association for Computing Machinery},
    address = {New York, NY, USA},
    url = {https://doi.org/10.1145/3511808.3557359},
    doi = {10.1145/3511808.3557359},
    booktitle = {Proceedings of the 31st ACM International Conference on 
Information &amp; Knowledge Management},
    pages = {1350–1359},
    numpages = {10},
    keywords = {dialogue generation, personality consistency, persona 
expanding},
    location = {Atlanta, GA, USA},
    series = {CIKM '22}
}
```

[Arxiv Link](https://arxiv.org/abs/2208.10816)
