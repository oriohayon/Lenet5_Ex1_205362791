# Lenet5 Deep Learning Seminar
### Author: Ori Ohayon
#### ID: 205362791

#### Overview
The project is implementing 4 types of LeNet 5 variations.
The model of the different CNNs was created using pytorch.

The different LeNet 5 implemented for train and test:
1. Regular - No regularization. A slight modified LeNet 5 CNN, which is similar to the LeNet5 proposed by LeCun et al. (1998).
2. DropOut - Variation that uses drop out method on the fully connected layers.
3. WeightDecay - Variation that uses weight decay method in the optimizer selection.
4. BatchNorm - Variation that uses batch normalization method.

#### Directories of data&results
* The model is using the FashionMNIST data set to train and test. The data set is downloaded (if not exists already) to directory 'data' in the local project location.
* The training results (parameters of the network) are saved to a pickle (.pt) in 'Saved_Train_Dict' directory under the local project location.
* Figures of LogLoss and Accuracy are saved to 'Figures' under target_dir directory.

### Pay Attention:
#### The LeNet5 saved weights and parameters are saved and load in a specific format. example:
'<target_dir>/Saved_Train_Dict/lenet_5_dropout_epoch_0.pt'
'<target_dir>/Saved_Train_Dict/lenet_5_dropout_epoch_1.pt'

.
.

### How to Train/Test the model
The call to the model is using the run.py in the terminal. 

definition:
python run.py lenet_mode run_mode n_epoch target_dir

Example:
python3 run.py regular train_test 30 "/content/drive/MyDrive/EX1_205362791"

The arguments can take the following values:
1. lenet_mode - define type of LeNet. use only: regular / dropout / weightdecay / batchnorm
2. run_mode - define what to run: train / test / train_test
3. n_epoch - number of epochs to test/train
4. target_dir - path to the directory we want to save results to.

