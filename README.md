# classification-using-MobileNetV2
## 1. PyTorch Implemention of MobileNet V2

Reproduction of MobileNet V2 architecture as described in MobileNetV2: Inverted Residuals and Linear Bottlenecks by Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov and Liang-Chieh Chen on ILSVRC2012 benchmark with PyTorch framework.

This implementation provides an example procedure of training and validating any prevalent deep neural network architecture, with modular data processing, training, logging and visualization integrated.

<br>
This is Pytorch implementation og MobileNetV2 architectue with dafault width = 1.0
<br>


## usage for this classification
1. Clone this repo to local
<br>
git clone https://github.com/ram-parvesh/classification-using-MobileNetV2.git

# Requirements
2. create conda environment using command 
<br>
create env -n mobileNetV2 python=3.7
<br>
conda activate mobileNetV2 ## <-- activate environment using command

## Required libraries
<br>
Python 3.7
<br>
numpy 1.21.6
<br>
scikit-image 0.19.3
<br>
python-opencv PIL 5.2.0
<br>
PyTorch 0.4.0
<br>
torchvision 0.13.1
<br>
glob

## install dependencies using command
3. pip install -r requirement.txt


4. Cd to the directory 'classification-using-MobileNetV2',run the train or inference process by command: python train.py or python test.py respectively.

<br>

[Training Notebook](https://github.com/ram-parvesh/classification-using-MobileNetV2/blob/master/train.ipynb)
<br>

[Test notebook](https://github.com/ram-parvesh/classification-using-MobileNetV2/blob/master/TesT.ipynb)
<br>

[Custom Datasets Notebook](https://github.com/ram-parvesh/classification-using-MobileNetV2/blob/master/CustomDataset.ipynb)
<br>
