# VCANet
Visual Chirality Attention (VCA) module

This is an implementation of VCANet, created by Ying Zheng.

## Our environments

Python: 3.7.1

OS: Ubuntu 18.04

CUDA: 10.1

Toolkit: PyTorch 1.4.0, torchvision 0.5.0

GPU: GTX 2080Ti

## Start Up

You can run the train.py to train the VCANet as follow:

```
python train.py
```

## Experiments

Sketch recognition on TU-Berlin dataset.

|Method|Backbone|#.Param.|FLOPs|Accuracy|
|:-:|:-:|:-:|:-:|:-:|
|ResNet|ResNet-50|25.56M|4.12G|77.3|
|SENet|ResNet-50|28.09M|4.13G|78.3|
|CBAM|ResNet-50|28.07M|4.12G|78.5|
|ECA-Net|ResNet-50|25.56M|4.12G|78.8|
|**VCANet**|ResNet-50|26.08M|4.12G|**79.7**|
|ResNet|ResNet-101|44.55M|7.84G|79.5|
|SENet|ResNet-101|49.33M|7.86G|79.6|
|CBAM|ResNet-101|49.30M|7.85G|79.7|
|ECA-Net|ResNet-101|44.55M|7.84G|80.0|
|**VCANet**|ResNet-101|45.07M|7.84G|**80.6**|
|ResNet|ResNet-152|60.19M|11.57G|80.3|
|SENet|ResNet-152|66.82M|11.59G|80.5|
|CBAM|ResNet-152|66.78M|11.58G|80.4|
|ECA-Net|ResNet-152|60.19M|11.57G|80.9|
|**VCANet**|ResNet-152|60.72M|11.57G|**81.2**|
|ResNet|ResNeXt-101 (32x8d)|88.79M|16.49G|80.5|
|SENet|ResNeXt-101 (32x8d)|93.57M|16.50G|81.0|
|CBAM|ResNeXt-101 (32x8d)|93.54M|16.49G|80.4|
|ECA-Net|ResNeXt-101 (32x8d)|88.79M|16.49G|80.8|
|**VCANet**|ResNeXt-101 (32x8d)|89.32M|16.49G|**81.5**|

## Todo

- [ ] add more pretrained models
- [ ] add more results

## Citation

```
@article{zheng2020vcanet,
  title={Visual Chirality Meets Freehand Sketches},
  author={Zheng, Ying and Zhang, Yiyi and Xu, Xiaogang and Wang, Jun and Yao, Hongxun},
  year={2020}
}
```

## Contact

If you have any questions or suggestions, please contact zhengyinghit@outlook.com. Thanks for your attention!
