# DeepRAG: Thinking to Retrieve Step by Step for Large Language Models

https://arxiv.org/abs/2502.01142


## Build Wikipedia index

Download the Wikipedia dump from the [DPR repository](https://github.com/facebookresearch/DPR/blob/main/dpr/data/download_data.py#L32) using the following command:

```bash
mkdir -p data/dpr
wget -O data/dpr/psgs_w100.tsv.gz https://dl.fbaipublicfiles.com/dpr/wikipedia_split/psgs_w100.tsv.gz
pushd data/dpr
gzip -d psgs_w100.tsv.gz
popd
```

Use Elasticsearch to index the Wikipedia dump:

```bash
cd data
wget -O elasticsearch-7.17.9.tar.gz https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-7.17.9-linux-x86_64.tar.gz  # download Elasticsearch
tar zxvf elasticsearch-7.17.9.tar.gz
rm elasticsearch-7.17.9.tar.gz 
cd elasticsearch-7.17.9
nohup bin/elasticsearch &  # run Elasticsearch in background
cd ../..
python prep_elastic.py --data_path data/dpr/psgs_w100.tsv --index_name wiki  # build index
```

## Training

If you want to construct data from scratch, you can following the instructions below. using Llama-3-8B as example:

### Stage I

#### 1. launch model

```bash
bash scripts/launch/run.sh

bash scripts/launch/run-72b.sh
```

#### 2. inference

```bash
bash scripts/data-construct/stage1.sh
```

Inference results will be saved in `construct/sft/*`

#### 3. filter

use eval script to evaluate the inference result in `construct/sft/*`.

use `scripts/data-construct/filter-stage1/extract.py` to filter the correct response. Then use `scripts/data-construct/filter-stage1/tokenize_sft.py` to tokenize the response for further training.
Meanwhile, you can use `scripts/data-construct/filter-stage1/reformat.py` to visualize the tokenize data format.

The tokenized data will be saved to `construct/sft/tokenized_dataset`.


#### 4. training

We use [Llama-Factory](https://github.com/hiyouga/LLaMA-Factory) for training.

Training scripts in `scripts/training/stage1.sh`


### Stage II

#### 1. launch model

```bash
# model path should be the trained sft model
bash scripts/launch/run.sh
```

#### 2. inference

```bash
bash scripts/data-construct/stage2.sh
```

Inference results will be saved in `construct/dpo/*`


#### 3. filter

use eval script to evaluate the inference result in `construct/dpo/*`.

use `scripts/data-construct/filter-stage2/tokenize_dpo.py` to tokenize the response for further training.
Meanwhile, you can use `scripts/data-construct/filter-stage2/make_pair.py` to visualize the tokenize data format.

The tokenized data will be saved to `construct/dpo/tokenized_dataset`.


#### 4. training

We use [Llama-Factory](https://github.com/hiyouga/LLaMA-Factory) for training.

Training scripts in `scripts/training/stage2.sh`


### Stage II with RL

We further validate our framework's effectiveness using reinforcement learning. The implementation can be found in our RL extension repository.

https://github.com/gxy-gxy/Search-R1-for-DeepRAG/tree/main 


## Inference

### 1. launch model


```bash
# modify with your own model path
bash scripts/launch/run.sh
```

### 2. inference

```bash
bash scripts/inference/run.sh
```

## Eval

eavl is inherient from [DRAGIN](https://github.com/oneal2000/DRAGIN).

```bash
bash scripts/eval/run.sh
```


## Model Checkpoints

comming soon

# Acknowledgment

This code is heavily based on [DRAGIN](https://github.com/oneal2000/DRAGIN), which provides a framework for building multiple baselines. We enhance the inference pipeline with API-based methods along with multi-process acceleration.


# Citation

If you find this work helpful, please cite our paper:

```
@article{guan2025deepragthinkingretrievalstep,
    title={DeepRAG: Thinking to Retrieve Step by Step for Large Language Models}, 
    author={Xinyan Guan and Jiali Zeng and Fandong Meng and Chunlei Xin and Yaojie Lu and Hongyu Lin and Xianpei Han and Le Sun and Jie Zhou},
    year={2025},
    journal={arXiv preprint arXiv:2502.01142},
    url={https://arxiv.org/abs/2411.11504}
}

```
