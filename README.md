# Pneumonia Detection

Our goal for this project is to develop and train a Convoluted Neural Network (CNN) model to correctly identify if
the image being viewed is of a patient with normal and healthy lungs or if the patient has pneumonia.

![Image Normal and Pneumonia](https://github.com/samjdedes/pneumonia_detection/blob/master/reports/visualizations/Screen%20Shot%202020-11-12%20at%204.52.59%20PM.png)
                               (Images are from dataset)


## TABLE OF CONTENTS

Our home repository contains the project environment and information about our project.

### Notebooks

[Exploratory Data Analysis](notebooks/exploratory) 

[Final Report Notebook](notebooks/report)

### Reports
[Executive Summary](reports/presentation)

[Visualizations](reports/visualizations)

### Data

[How to access data](data)


### SRC

[Custom Functions](src)

### ReadMe

[Read Me](README.md)

## Project Goal and Background

We used the Cross-Industry Standard Process for Data Mining(CRISP-DM) approach to this project. 

## Business Understanding 

According to [Mayo Clinic](https://www.mayoclinic.org/diseases-conditions/pneumonia/symptoms-causes/syc-20354204), "Pneumonia is an infection that inflames the air sacs in one or both lungs. The air sacs may fill with fluid or pus (purulent material), causing cough with phlegm or pus, fever, chills, and difficulty breathing. A variety of organisms, including bacteria, viruses and fungi, can cause pneumonia." Particularly for children, pneumonia can be very dangerous. 

"Pneumonia is the single largest cause of death in children worldwide. Every year, it kills an estimated 1.4 million children under the age of five years, accounting for 18% of all deaths of children under five years old worldwide." These statistics from the [World Health Organization](https://www.who.int/maternal_child_adolescent/news_events/news/2011/pneumonia/en/) show the importance of an accurate and timely diagnosis of the infection. 

There is often a [backlog](https://schwanerinjury.com/delayed-mri-ct-scan-or-x-ray-can-death-injury/) of x-ray results due to a shortage of radiologists. Some hospitals have tried to address this problem by outsourcing x-rays to other countries in different time zones to have more coverage. 

This neural network does not mean to replace any radiologists' inspection of an x-ray. If a child is getting a chest x-ray it is likely already considered a priority case. Using this neural netowork could help doctors and hospital administrators think about ways to help their patient prioritization protocols, how to possibly expedite the process for the pedriatric patients with suspected pneumonia, and whether or not it is helpful to them. 

The stakeholders here are the doctors treating pediatric patients with suspected pneumonia. 

## Data Understanding

The dataset is comprised of x-ray images of the frontal plane, anterior to posterior, of pediatric patients. Within these, 1,583 show normal lungs, and 4,273 show lungs with pneumonia. The original dataset comes from Kermany et al. on [Mendeley](https://data.mendeley.com/datasets/rscbjbr9sj/2). The Mendeley dataset includes a second set of images--ocular--that are outside of the scope of this project so we opted to use the [Kaggle](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia) dataset that already has just the relevant x-rays. 

Having a high recall score for this type of neural network is important because a false negative is much more harmful than a false positive. 

## Data Preparation
After retrieving the data from the dataset it is important to classify images according to their labels. Because we are running a CNN, a supervised learning machine method, labels need to be attributed to images to help in the process of reducing loss, and increasing recall and accuracy. After this is done, each image is converted into an 1-D array, and divided by 255. The values of the grayscale pixels range from 0-255. Dividing each pixel by 255 normalizes our grayscale to values between 0-1, and also helps our CNN algorithm converge faster.

![class imbalance](https://github.com/samjdedes/pneumonia_detection/blob/PR_branch/visualization/image_classification.png)
Next we needed to address our class imbalance. The first was to use a keras preprocessing function called ImageDataGenerator. ImageDataGenerator takes a batch of images used for training, applies a series of random transformation to each image in the batch (including random rotation, resizing and shearing), and replaces the original batch with the new randomly transformed batch. This effectively allows us to expand the training dataset in order to improve the performance and ability of the model to generalize.



## Modeling

Artificial intelligence (AI) Deep Neural Network makes it possible to process images in the form of pixels as input and to predict the desired classification as output. The development of Convolutional Neural Network (CNN) layers trains a model for significant gains in the ability to classify images and detect objects in a picture.  Multiple processing layers use image analysis filters, or convolutions as the model is trained.

The convolutional layers in the network along with filters help in extracting the spatial and temporal features in an image. The layers have a weight-sharing technique, which helps in reducing computation efforts. 

A Convolution Neural Network (CNN) is built on three broad strategies: 

  1)Learn features using Convolution layer 
  
  2)Reduce computational costs by down sample the image and reduce dimensionality using Max-Pooling(subsampling)
  
  3)Fully connected layer to equip the network with classification capabilities
  
  
![CNN graph by PR](https://github.com/samjdedes/pneumonia_detection/blob/PR_branch/visualization/cnn.jpg)

The performance of the model is evaluated using Recall and Accuracy metrics calculates how many of the Actual Positives our model capture through labeling it as Positive (True Positive). Recall calculates how many of the Actual Positives our model capture through labeling it as Positive (True Positive), in other words recall means the percentage of a pneumonia correctly identified. More accurate model lead to make better decision. The cost of errors can be huge but optimizing model accuracy mitigates that cost. 

The best model from this study achieved a Recall score of 93%, while scoring 90% of overall Accuracy.

![recall/accuracy](https://github.com/samjdedes/pneumonia_detection/blob/master/visualization/Model1_Epoch12_Batch32%20%20model_accuracy_recall.png)


![lime visuals](https://github.com/samjdedes/pneumonia_detection/blob/master/visualization/Sample_True_Predictions_5.png)


## Evaluation

Our best model had a final recall score of 93% and an overall accuracy score of 91%, this means that our model is correctly identifying an overwhelming majority of the x-ray images. This is a great start, but the fact still remains that there are still patients that have pneumonia but are being misclassified as having healthy and normal lungs. A few theories why this may be is that we have a relatively small dataset in regards to how many of our images are classified as normal. Also it may be difficult to further increase this score due to the nature that those who are coming in for chest x-rays are not doing so on a whim. They may be experiencing difficulty breathing any cardiovascular difficulties. It is not immediately clear if these patients were excluded from the study. So it is not apparent if the model is struggling with classification because even though a patient is identified as healthy, they may still have some abnormality present in their chest cavity.  

Another limitation our model has is a lack of data. There is currently difficulty gaining access to publicaly available pediatric chest x-ray images. We would have liked to include additional data, but in our search for more, we were unable to find x-ray images from children with pneumonia and/or with normal lungs. There are datasets available with adult lung images, however these would not be a good test set because adult and pediatric chest cavity structures are different and the model would not be able to accurately predict. A potential additional step, that is outside of the scope of this project, is to add adult to training, test, and validation sets. 




## Potential Next Steps

Our next steps would be to divide our images classified as pneumonia between viral pneumonia, and bacterial pneumonia. We would effectively be adding another class that our model would need to classify. But we thought by splitting our pneumonia classes we may help our model by leveling out our class imbalance. Further we believe that if our model was succesful at identifying all three classes, it may aid medical practitioners to implement the appropriate medicinal therapies for the respective diagnoses of patients with viral pneumonia and bacterial pneumonia.
