# Training ResNet50 on ImageNet-1k


## Dataset details

ImageNet-1k is a subset of the ImageNet dataset, containing 1000 classes with 1.2 million images. 

Kaggle link to the dataset: https://www.kaggle.com/datasets/c/imagenet-object-localization-challenge


Check [README.md](./utils/README.md) in the utils folder for more details.


## Model details

ResNet50 is a convolutional neural network that is 50 layers deep. It is a variant of the ResNet architecture, which is known for its depth and accuracy. ResNet50 has 49 layers in total, including the input layer, the output layer, and the convolutional layers in between.

Model loaded without pretrained weights.


## How to run

```
pip install -r requirements.txt
```

```bash
python src/train.py
```
