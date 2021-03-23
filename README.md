# RVL-BERT
This repository accompanies our IEEE Access paper "[Visual Relationship Detection with Visual-Linguistic Knowledge from Multimodal Representations](https://arxiv.org/abs/2009.04965)" and contains validation experiments code and the models on the SpatialSense and the VRD dataset.


## Installation
```
conda create -n rvl-bert python=3.6 scipy numpy
conda activate rvl-bert
# Go to https://pytorch.org/ to install pytorch=1.1.0 before proceeding
# Can run: conda install pytorch=1.1.0 cudatoolkit=9.0 -c pytorch
pip install Cython
pip -r requirements.txt
./scripts/init.sh
```


## Dataset
### SpatialSense
1. Download SpatialSense dataset from https://drive.google.com/drive/folders/125fgCq-1YYfKOAxRxVEdmnyZ7sKWlyqZ 
2. Put the files under `$RVL_BERT_ROOT/data/spasen` and unzip the `images.tar.gz` as `images/` there. Ensure there're two folders (`flickr/` and `nyu`) below `$RVL_BERT_ROOT/data/spasen/images/`.

### VRD
1. Download the VRD dataset [images](http://imagenet.stanford.edu/internal/jcjohns/scene_graphs/sg_dataset.zip) and [annotations](https://drive.google.com/file/d/16hU27W3T5Df7sPmioE-jPdJAlpMS3V3x/view?usp=sharing)
2. Put the `sg_train_images/` and `sg_test_images/` folders under `$RVL_BERT_ROOT/data/vrd/images`.
3. Put all `.json` files under `$RVL_BERT_ROOT/data/vrd/`.


## Checkpoints & Pretrained Weights
### Common
Download the pretrained weights from https://drive.google.com/file/d/1vEPOmQwFuBpJaO2ek5bC_rl2CLkgaxEk/view?usp=sharing and put the `pretrained_model/` folder under `$RVL_BERT_ROOT/model/`.

### SpatialSense
Download the trained checkpoint from https://drive.google.com/file/d/1G26POFKTSrkxl1STdPi4a-Rqrphdt8oC/view?usp=sharing and put the `.model` file under `$RVL_BERT_ROOT/checkpoints/spasen/`.

### VRD
Download the trained checkpoints and put the `.model` files under `$RVL_BERT_ROOT/checkpoints/vrd/`:
- [Basic model](https://drive.google.com/file/d/1mx8aDVGm4HoGp5zcCmCziLJbJxC93WhH/view?usp=sharing)
- [Basic + VL](https://drive.google.com/file/d/10d1Ghf2LsjwPLxfaJcheKbz7DMpZrru5/view?usp=sharing)
- [Basic + VL + S](https://drive.google.com/file/d/1NQImpyp5Ddi66SUSs9TH2r1NO0uLU8Yh/view?usp=sharing)
- [Full model](https://drive.google.com/file/d/1G_QnD3hI-M8O0-k1PqvdP9oBul7thCze/view?usp=sharing)


## Validation
Run the following commands to reproduce experiment results. A single GPU (NVIDIA Quadro RTX 6000, 24G memory) is used by default.

### SpatialSense
- Full model
```
python spasen/test.py --cfg cfgs/spasen/full-model.yaml --ckpt checkpoints/spasen/full-model-e44.model --bs 8 --gpus 0 --model-dir ./ --result-path results/ --result-name spasen_full_model --split test --log-dir logs
```

### VRD
- Basic model: 
```
python vrd/test.py --cfg cfgs/vrd/basic.yaml --ckpt checkpoints/vrd/basic-e59.model --bs 1 --gpus 0 --model-dir ./ --result-path results/ --result-name vrd_basic --split test --log-dir logs/
```

- Basic model + Visual-Linguistic Commonsense Knowledge
```
python vrd/test.py --cfg cfgs/vrd/basic_vl.yaml --ckpt checkpoints/vrd/basic-vl-e59.model --bs 1 --gpus 0 --model-dir ./ --result-path results/ --result-name vrd_basic_vl --split test --log-dir logs/
```

- Basic model + Visual-Linguistic Commonsense Knowledge + Spatial Module
```
python vrd/test.py --cfg cfgs/vrd/basic_vl_s.yaml --ckpt checkpoints/vrd/basic-vl-s-e59.model --bs 1 --gpus 0 --model-dir ./ --result-path results/ --result-name vrd_basic_vl --split test --log-dir logs/
```

- Full model
```
python vrd/test.py --cfg cfgs/vrd/basic_vl_s_m.yaml --ckpt checkpoints/vrd/basic-vl-s-m-e59.model --bs 1 --gpus 0 --model-dir ./ --result-path results/ --result-name vrd_basic_vl --split test --log-dir logs/
```

# Credit
This repository is mainly based on [VL-BERT](https://github.com/jackroos/VL-BERT).

# Citation
Please cite our paper if you find the paper or our code help your research!
```
@article{chiou2020rvl,
  title={RVL-BERT: Visual Relationship Detection with Visual-Linguistic Knowledge from Pre-trained Representations},
  author={Chiou, Meng-Jiun and Zimmermann, Roger and Feng, Jiashi},
  journal={arXiv preprint arXiv:2009.04965},
  year={2020}
}
```
