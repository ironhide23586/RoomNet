# RoomNet
A Convolutional Neural Net to classify pictures of different rooms of a house/apartment with 88.9 % validation accuracy over 1839 images.

Full presentation at - https://github.com/ironhide23586/RoomNet/raw/master/documentation/RoomNet%20Presentation.pptx

This is a custom neural net I designed to classify an input image in to one of the following 6 classes (in order of their class IDs) -
* Backyard
* Bathroom
* Bedroom
* Frontyard
* Kitchen
* LivingRoom

## Architecture -

### Building Blocks -

<img src="https://github.com/ironhide23586/RoomNet/blob/master/documentation/conv_block.png" alt="drawing" width="300"/> <img src="https://github.com/ironhide23586/RoomNet/blob/master/documentation/residual_block.png" alt="drawing" width="300"/>
<img src="https://github.com/ironhide23586/RoomNet/blob/master/documentation/dense_block.png" alt="drawing" width="300"/>

### Full Network Architecture -

<img src="https://github.com/ironhide23586/RoomNet/blob/master/documentation/network.png" alt="drawing" width="300"/>

## Out-of-box Inference -

Optimized inference code in **infer.py**. Refer to the short code in the *main* method calling the *classify_im_dir* method.

## Training -

* Input image size = 224 x 224 (tried 300 x 300, 600 x 600)
* Softmax Cross Entropy Loss used with L2 Weight normalization
* Dropout varied from 0 (initially) to 0.3 (intermittently near the end of training). Dropout layers placed after every block.
* Batch Normalization moving means & vars were frozen when being trained with dropout
* Adam Optimizer used with exponential learning rate decay.
* Initially trained with in-batch computation of BatchNorm moving means/vars. Followed this by training net, by disabling this computation and using frozen means/vars during training. Resulted in 10% immediate jump in validation accuracy.
* Batch Size varied from 8 (in the beginning) to 45 (towards training end) as – 8 -> 32 -> 40 -> 45
* Asynchronous Data Reader designed with a Queue based architecture which allows for quick data I/O during training even with large batch sizes.

## Conversion to Inference Optimized Version -

* Discarded all back propagation/training related compute node from the Tensorflow Graph.
* Model size reduced from ~2 MB to ~800 KB.
* *network.py* contains class defining the model called “RoomNet”
* Output is an excel file mapping each image path to its label. There is also provision to split an input directory to directories corresponding to the class names and automatically fill the relevant image in its respective directory.

## Training Environment -

* Training done using Tensorlfow + CUDA 10.0 + cuDNN on NVIDIA GTX 1070 laptop grade GPU with 8GB of GPU memory
* Compute system used is an Alienware m17 r4.
* CPU used is an Intel Core i7 – 6700HQ with 8 logical cores at 2.6 GHz of base speed (turbo boost to ~3.3 GHz)
* Number of training steps from scratch to reach best model is 157,700.
* Time spent on training - ~48 hours

## Previous Approaches tried -

* Tried training the final dense NASnet mobile but accuracy never crosses 60%.
* Tried the same with InceptionV3 but convergence takes too damn long.

## Performance Plots -
#### Validation Accuracy
<img src="https://github.com/ironhide23586/RoomNet/blob/master/documentation/performance_plots/accuracy_plot.png" alt="drawing" width="500"/>

#### Validation Class-wise F-Score
<img src="https://github.com/ironhide23586/RoomNet/blob/master/documentation/performance_plots/fscore_plot.png" alt="drawing" width="800"/>

#### Validation Class-wise Precision
<img src="https://github.com/ironhide23586/RoomNet/blob/master/documentation/performance_plots/precision_plot.png" alt="drawing" width="800"/>

#### Validation Class-wise Recall
<img src="https://github.com/ironhide23586/RoomNet/blob/master/documentation/performance_plots/recall_plot.png" alt="drawing" width="800"/>
