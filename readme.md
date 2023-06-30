# Surface Defect Classification in Carbon Look Components Using Deep Neural Networks

This repository contains the source code of the experiments presented in the paper

>A. Silenzi, S. Tomassini, N. Falcionelli, P. Contardo, A. Bonci, A.F. Dragoni, P. Sernani, *Quality Control of Carbon Look Components via Surface Defect Classification with Deep Neural Networks*.

The paper is currently under review for the publication in the [Sensors MDPI journal](https://www.mdpi.com/journal/sensors).

Specifically, the experiments are **accuracy tests of ten different 2D Convolutional Neural Networks (2D CNNs) pretrained on Imagenet**. The models are combined with fully connected layers to classify images of carbon-look components into **defective and non-defective** and to recognize **different types of surface defects**. Such models perform a classification on samples of a real case study, collected into two different datasets to test binary classification (negative, i.e. no defects, vs positive, i.e. with defects) and multi-class classification (negative, i.e. no defects, vs non-recoverable defects vs recoverable defects).

The source code implementing the ten models and the experiments on the datasets is contained in a Jupyter notebook, available in the “notebooks” directory of this repository. The notebook is "[Surface_Defect_Classification_in_Carbon_Look_Components_Using_Deep_Neural_Networks.ipynb](notebooks/Surface_Defect_Classification_in_Carbon_Look_Components_Using_Deep_Neural_Networks.ipynb)"

All the experiments ran on GPU, using and Keras 2.6.0, with the TensorFlow 2.6.0 backend, and scikit-learn 1.0.2.

## Data Description

The classification experiments are based on a database of carbon look component images. The database of images is available in the following public repository: 
> <https://github.com/airtlab/surface-defect-classification-in-carbon-look-components-dataset>

The images are 224 x 224 pixels (96 ppi) in JPEG format. The images were originally taken at 3968 x 2976 pixels, then cropped at the center and resized at 224 x 224. The database is split into two datasets:

- One dataset is binary, including two classes. 200 images are labeled as “no defect” as they are with no defects or present limited recoverable porosities;  200 images are labeled as “defect” as they include weft discontinuities. This dataset is intended for binary classification.

- The second dataset is multi-class, including three classes, i.e. “no defect” (negative), with recoverable defects, and with non-recoverable defects. The dataset contains 1500 images, 500 per class. The “no defect” class includes images of components without any surface defect. The recoverable defect class includes images of components with limited porosities and the infiltration of external materials (such as aluminum). Such defects can be treated and corrected. Finally, the non-recoverable defect class includes images of components with weft discontinuites, severe porosities, and resin accumulations.

## Experiments 

Ten different deep neural networks were implemented and tested on both the available datasets. Each neural network is a combination of a pretrained CNN and fully connected layers to perform the classification.

The models are combined with fully connected layers to classify images of carbon-look components into defective and non-defective and to recognize different types of surface defects. Such models perform a classification of the images included in the two datasets.

The tested 2D CNNs are:

- VGG16 ([https://keras.io/api/applications/vgg/#vgg16-function](https://keras.io/api/applications/vgg/#vgg16-function))
- VGG19 ([https://keras.io/api/applications/vgg/#vgg19-function](https://keras.io/api/applications/vgg/#vgg19-function))
- ResNet50V2 ([https://keras.io/api/applications/resnet/#resnet50v2-function](https://keras.io/api/applications/resnet/#resnet50v2-function))
- ResNet101V2 ([https://keras.io/api/applications/resnet/#resnet101v2-function](https://keras.io/api/applications/resnet/#resnet101v2-function))
- ResNet152V2 ([https://keras.io/api/applications/resnet/#resnet152v2-function](https://keras.io/api/applications/resnet/#resnet152v2-function))
- InceptionV3 ([https://keras.io/api/applications/inceptionv3](https://keras.io/api/applications/inceptionv3))
- MobileNetV2 ([https://keras.io/api/applications/mobilenet/#mobilenetv2-function](https://keras.io/api/applications/mobilenet/#mobilenetv2-function))
- NASNetMobile ([https://keras.io/api/applications/nasnet/#nasnetmobile-function](https://keras.io/api/applications/nasnet/#nasnetmobile-function))
- DenseNet121 ([https://keras.io/api/applications/densenet/#densenet121-function](https://keras.io/api/applications/densenet/#densenet121-function))
- Xception ([https://keras.io/api/applications/xception/](https://keras.io/api/applications/xception/))

The experimental evaluation is based on the Stratified Shuffle Split cross-validation scheme to do ten randomized splits of the available data in 80% for training and 20% for testing. With the two end-to-end models the 12.5% of the training data (i.e. 10% of the entire dataset) was use for validation. For each split the confusion matrix and a classification report are printed as output. Moreover, at the end of each test, the average value of accuracy, precision, recall, and F1-score are reported, as well as the ROC computed in each split.

Each model was trained in a end-to-end fashion on the proposed datasets, fine tuning the pretrained CNNs and training from scratch the added fully connected layers. The best hyperparameters in terms of classification accuracy on the binary dataset and the multi-class dataset were selected. Specifically, the following parameters were tested:

- The number of final layers to be fine tuned on the proposed dataset, testing 8, 4, and 0.

- The optimizer to perform the error backpropagation, testing the Stochastic Gradient Descent (SGD), with 0.9 momentum, and Adam. For different learning rates were tested, i.e. 0.001, and 0.0001.

- The use of Batch Normalization for regularization between the Global Average Pooling and the first dense layer.

- The number of fully connected layers to be added to the pre-trained CNNs to perform the final classification, testing a single dense layer composed of 512 ReLU neurons followed by a final layer with Softmax activation, and two dense layers composed of 256 and 128 ReLu layers, followed by the Softmax layer. The following tables include the best combination of hyperparameters for each model, on the binary dataset and on the multi-class dataset.

### Best hyperparameters on the binary dataset

|                  | **Fine Tuning** | **Optimizer** | **Learning Rate** | **Batch Normalization** | **Dense 512 Units** | **Dense 256 Units** | **Dense 128 Units** |
|:----------------:|:---------------:|:-------------:|:-----------------:|:-----------------------:|:-------------------:|:-------------------:|:-------------------:|
|     **VGG16**    |      Last 8     |      SGD      |       0.0001      |            -            |          x          |          -          |          -          |
|     **VGG19**    |      Last 8     |      SGD      |       0.001       |            -            |          x          |          -          |          -          |
|  **ResNet50V2**  |      Last 8     |      Adam     |       0.0001      |            -            |          x          |          -          |          -          |
|  **ResNet101V2** |      Last 8     |      Adam     |       0.001       |            -            |          x          |          -          |          -          |
|  **ResNet152V2** |      Last 8     |      Adam     |       0.0001      |            -            |          x          |          -          |          -          |
|  **InceptionV3** |      Last 8     |      SGD      |       0.001       |            -            |          x          |          -          |          -          |
|  **MobileNetV2** |        0        |      Adam     |       0.001       |            -            |          -          |          x          |          x          |
| **NASNetMobile** |      Last 4     |      Adam     |       0.0001      |            -            |          -          |          x          |          x          |
|  **DenseNet121** |      Last 8     |      Adam     |       0.0001      |            -            |          x          |          -          |          -          |
|   **Xception**   |      Last 4     |      Adam     |       0.0001      |            -            |          -          |          x          |          x          |

### Best hyperparameters on the multi-class dataset

|                  | **Fine Tuning** | **Optimizer** | **Learning Rate** | **Batch Normalization** | **Dense 512 Units** | **Dense 256 Units** | **Dense 128 Units** |
|:----------------:|:---------------:|:-------------:|:-----------------:|:-----------------------:|:-------------------:|:-------------------:|:-------------------:|
|     **VGG16**    |      Last 8     |      SGD      |       0.001       |            x            |          x          |          -          |          -          |
|     **VGG19**    |      Last 8     |      SGD      |       0.001       |            x            |          x          |          -          |          -          |
|  **ResNet50V2**  |      Last 8     |      Adam     |       0.001       |            -            |          x          |          -          |          -          |
|  **ResNet101V2** |      Last 8     |      Adam     |       0.001       |            -            |          x          |          -          |          -          |
|  **ResNet152V2** |      Last 8     |      Adam     |       0.0001      |            -            |          x          |          -          |          -          |
|  **InceptionV3** |      Last 8     |      Adam     |       0.0001      |            -            |          x          |          -          |          -          |
|  **MobileNetV2** |      Last 8     |      SGD      |       0.0001      |            -            |          x          |          -          |          -          |
| **NASNetMobile** |      Last 8     |      Adam     |       0.0001      |            -            |          x          |          -          |          -          |
|  **DenseNet121** |      Last 8     |      Adam     |       0.0001      |            -            |          x          |          -          |          -          |
|   **Xception**   |      Last 8     |      Adam     |       0.0001      |            -            |          x          |          -          |          -          |
