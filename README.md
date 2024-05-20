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


## First Stage Training

After preparing the training data for MFIF and MEIF tasks, use 
```
python train_stage1.py
```
to start the training of the model.

## Second Stage Training

After preparing the training data for MFIF, MEIF, and IVF tasks, use 
```
python train_stage2.py
```
to start the training of the model

## Testing

After preparing the testing data, use
```
python test.py
```
## Citation
If you use our work, please consider citing:

  @article{wang2024general,
    title={A general image fusion framework using multi-task semi-supervised learning},
    author={Wang, Wu and Deng, Liang-Jian and Vivone, Gemine},
    journal={Information Fusion},
    pages={102414},
    year={2024},
    publisher={Elsevier}}
## Contact
Should you have any questions, please contact 947658333@qq.com
