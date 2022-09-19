# Environment
See [environment.yml](./environment.yml) for package requirements.   
We use NVIDIA GeForce RTX 3090 (24GB) to conduct our experiments.

# Download Dataset
The NIPS 2017 ImageNet-Compatible dataset is from [clevenhans](https://github.com/cleverhans-lab/cleverhans/tree/master/cleverhans_v3.1.0/examples/nips17_adversarial_competition/dataset). But we download this dataset from [Targeted-tranfer](https://github.com/ZhengyuZhao/Targeted-Tansfer).

# Some Variables
The variables OPT_PATH and NIPS_DATA in [utils.py](./utils.py) need to be specified with your own output path and the downloaded dataset path.

# Run our experiments
For performing T-Aug,
```
python transforms_main.py --gpu {gpu} --no-subset --white_box {model} --loss_fn {loss_fn} --attack SingleTransormsAttack --target --transform_type TargetAdvAugment --num_ops 5 --magnitude 2 --delete_ops Crop DI Rotate --file_tailor exp-taug-table
```
For performing H-Aug,
```
python transforms_main.py --gpu {gpu} --no-subset --white_box {model} --loss_fn {loss_fn} --attack SingleTransormsAttack --target --transform_type TargetAdvAugment --num_ops 2 --magnitude 8 --delete_ops ShearX ShearY TranslateX TranslateY Flip --file_tailor exp-haug-table
```
For performing T-Aug-TM,
```
python transforms_main.py --gpu {gpu} --no-subset --white_box {model} --loss_fn {loss_fn} --attack SingleTransormsAttack_TMI --target --transform_type TargetAdvAugment --num_ops 5 --magnitude 2 --delete_ops Crop DI Rotate --file_tailor exp-taug_tm-table
```
For performing H-Aug-TM,
```
python transforms_main.py --gpu {gpu} --no-subset --white_box {model} --loss_fn {loss_fn} --attack SingleTransormsAttack_TMI --target --transform_type TargetAdvAugment --num_ops 2 --magnitude 8 --delete_ops ShearX ShearY TranslateX TranslateY Flip --file_tailor exp-haug_tm-table
```
where {gpu} is the id of gpu device, {model} denotes the white-box model, and is selected from ['densenet121', 'inception_v3', 'resnet50', 'vgg16_bn'], {loss_fn} is selected from ['CE', 'Logit'].

For performing T-Aug in the ensemble-model transfer scenario,
```
python ensemble_main.py --gpu {gpu} --black_box {model} --loss_fn {loss_fn} --attack Ensemble_DTMI_Transforms --target --no-saveperts --num_ops 5 --magnitude 2 --delete_ops Crop DI Rotate --file_tailor exp-taug_ensemble-table --batch_size 5
```
For performing H-Aug in the ensemble-model transfer scenario,
```
python ensemble_main.py --gpu {gpu} --black_box {model} --loss_fn {loss_fn} --attack Ensemble_DTMI_Transforms --target --no-saveperts --num_ops 2 --magnitude 8 --delete_ops ShearX ShearY TranslateX TranslateY Flip --file_tailor exp-haug_ensemble-table --batch_size 5
```
where [model] denotes the attacked black-box model.

# Explore different $N$ and $M$
For T-Aug,
```
python transforms_main.py --gpu {gpu} --subset --white_box {model} --loss_fn {loss_fn} --attack SingleTransormsAttack --target --transform_type TargetAdvAugment --num_ops {N} --magnitude {M} --delete_ops Crop DI Rotate --file_tailor exp-taug-table-{N}_{M}
```
For H-Aug,
```
python transforms_main.py --gpu {gpu} --subset --white_box {model} --loss_fn {loss_fn} --attack SingleTransormsAttack --target --transform_type TargetAdvAugment --num_ops {N} --magnitude {M} --delete_ops ShearX ShearY TranslateX TranslateY Flip --file_tailor exp-haug-table-{N}_{M}
```

# Motivation
### Figure 1
```python
python motivation_run.py
```
### Figure 2
Attention heatmaps w.r.t. the target class in different models:
```python
python gradcam_figure2.py 
```
### Figure 3
IoUs of DI and other image transformations.
```python
python gradcam_ious_run.py
```
