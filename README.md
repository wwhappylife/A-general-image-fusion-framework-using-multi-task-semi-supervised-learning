# A-general-image-fusion-framework-using-multi-task-semi-supervised-learning
> **Abstract:** *Existing image fusion methods primarily focus on solving single-task fusion problems, overlooking the potential
information complementarity among multiple fusion tasks. Additionally, there has been no prior research in
the field of image fusion that explores the mixed training of labeled and unlabeled data for different fusion
tasks. To address these gaps, this paper introduces a novel multi-task semi-supervised learning approach to
construct a general image fusion framework. This framework not only facilitates collaborative training for
multiple fusion tasks, thereby achieving effective information complementarity among datasets from different
fusion tasks, but also promotes the (unsupervised) learning of unlabeled data via the (supervised) learning
of labeled data. Regarding the specific network module, we propose a so-called pseudo-siamese Laplacian
pyramid transformer (PSLPT), which can effectively distinguish information at different frequencies in source
images and discriminatively fuse features from distinct frequencies. More specifically, we take datasets of
four typical image fusion tasks into the same PSLPT for weight updates, yielding the final general fusion
model. Extensive experiments demonstrate that the obtained general fusion model exhibits promising outcomes
for all four image fusion tasks, both visually and quantitatively. Moreover, comprehensive ablation and
discussion experiments corroborate the effectiveness of the proposed method.* 
<hr />


## Training

After preparing the training data in ```data/``` directory, use 
```
python train.py
```
to start the training of the model. Use the ```de_type``` argument to choose the combination of degradation types to train on. By default it is set to all the 3 degradation types (noise, rain, and haze).

Example Usage: If we only want to train on deraining and dehazing:
```
python train.py --de_type derain dehaze
```

## Testing

After preparing the testing data in ```test/``` directory, place the mode checkpoint file in the ```ckpt``` directory. The pretrained model can be downloaded [here](https://drive.google.com/file/d/1wkw5QCQyM2msQOpV-PL2uag3QLs8jYFc/view?usp=sharing). To perform the evalaution use
```
python test.py --mode {n}
```
```n``` is a number that can be used to set the tasks to be evaluated on, 0 for denoising, 1 for deraining, 2 for dehaazing and 3 for all-in-one setting.

Example Usage: To test on all the degradation types at once, run:

```
python test.py --mode 3
```

## Citation
If you use our work, please consider citing:

    @article{wang2024general,
  title={A general image fusion framework using multi-task semi-supervised learning},
  author={Wang, Wu and Deng, Liang-Jian and Vivone, Gemine},
  journal={Information Fusion},
  pages={102414},
  year={2024},
  publisher={Elsevier}
}

