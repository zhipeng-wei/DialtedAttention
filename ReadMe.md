<div align="center">

<h1><a href="">What Do Visual Models Look At? Dilated Attention for Targeted Transferable Attacks</a></h1>

**International Journal of Computer Vision**

**[Zhipeng Wei](https://zhipeng-wei.github.io/), [Jingjing Chen](https://fvl.fudan.edu.cn/people/jingjingchen), [Yu-Gang Jiang](https://fvl.fudan.edu.cn/people/yugangjiang/)**

</div>

# Environment
See [environment.yml](./environment.yml) for package requirements.

# Download Dataset
The NIPS 2017 ImageNet-Compatible dataset is from [clevenhans](https://github.com/cleverhans-lab/cleverhans/tree/master/cleverhans_v3.1.0/examples/nips17_adversarial_competition/dataset). We download this dataset from [Targeted-tranfer](https://github.com/ZhengyuZhao/Targeted-Tansfer).

# Some Variables
The variables OPT_PATH and NIPS_DATA in [utils.py](./utils.py) need to be specified with your own output path and the downloaded dataset path.

# Run our experiments
## Our Attack
```
python main.py --gpu 0 --white_box resnet50 --attack DTMI_DilatedAttention --loss_fn Attention --target --baseline_cmd DI_TI_MI --linear_aug Ours --layers layer1 layer2 layer3 layer4 --step
```
In this command:
* --white_box resnet50 spcifies the white-box surrogate model
* --attack DTMI_DilatedAttention refers to the attack class we implemented in transfer_attacks.py.
* --loss_fn Attention specifies the use of our proposed Attention loss.
* --target means the transferable targeted attack.
* --baseline_cmd DI_TI_MI indicates that the attack integrates DI, TI, and MI techniques.
* --linear_aug Ours enables our dynamic linear augmentation method.
* --layers layer1 layer2 layer3 layer4 specifies the layers on which the attention loss is calculated.
**Note:** The specific layers used to compute attention loss vary by model:

- **densenet121**:  
  `features_denseblock1`, `features_denseblock2`, `features_denseblock3`, `features_norm5`

- **inception_v3**:  
  `Conv2d_4a_3x3`, `Mixed_5d`, `Mixed_6e`, `Mixed_7c`

- **vgg16_bn**:  
  `features_12`, `features_22`, `features_32`, `features_42`

## Ensemble Attack
```
python ensemble_main.py --gpu 0 --black_box densenet121 --attack Ensemble_CommonWeakness --Ours
```
In this command:
* --black_box densenet121 specifies that all models except Densenet121 are used as white-box models.
* --attack Ensemble_CommonWeakness indicates that the attack integrates the Common Weakness technique.
* --Ours enables our proposed method.

