# DU-GFER

# Towards Domain Generalization in Facial Expression Recognition: Debiasing and Uniformizing Domain-Invariant Space
## Data Preparation
   * Downloading the original images after obtaining official authorization for the mentioned datasets: [RAF-DB](http://whdeng.cn/RAF/model1.html), [SFEW](https://users.cecs.anu.edu.au/~few_group/AFEW.html), [Affectnet](http://mohammadmahoor.com/affectnet/), [FERplus](https://github.com/gitshanks/fer2013), [MMA](https://www.kaggle.com/datasets/mahmoudima/mma-facial-expression).
   * Allocating training and testing datasets.

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Train/test

To train/test the model(s) in the paper, run this command:

```train
python main.py
```


## Acknowledgment
1. [Zhang Y, Zheng X, Liang C, et al. Generalizable facial expression recognition[C]//European Conference on Computer Vision. Cham: Springer Nature Switzerland, 2024: 231-248.](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/02196.pdf)

