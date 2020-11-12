# Pneumonia Detection

Our goal for this project is to develop and train a Convoluted Neural Network (CNN) model to correctly identify if
the image being viewed is of a patient with normal and healthy lungs or if the patient has pneumonia.

![Image Normal](https://github.com/samjdedes/pneumonia_detection/blob/master/reports/visualizations/Screen%20Shot%202020-11-10%20at%2011.47.47%20AM.png)
                               (Images are from dataset. Left is bacterial pneumonia, right is a normal lung)


## TABLE OF CONTENTS

Our home repository contains the project environment and information about our project.

### Notebooks

[Exploratory Data Analysis](exploratory) 

[Final Report Notebook](report)

### Reports
[Executive Summary](presentation)

[Visualizations](visualizations)

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

This neural network does not mean to replace any radiologists' inspection of an x-ray. If a child is getting a chest x-ray it is likely already considered a priority case.  Using this neural netowork could help doctors and hospital administrators think about ways to help their patient prioritization protocols, how to possibly expedite the process for the pedriatric patients with suspected pneumonia, and whether or not it is helpful to them. 

The stakeholders here are the doctors treating pediatric patients with suspected pneumonia. 

## Data Understanding

The dataset is comprised of greyscale x-ray images of the frontal plane, anterior to posterior, of pediatric patients . Within these, 1,583 show normal lungs, and 4,273 show lungs with pneumonia. The original dataset comes from Kermany et al. on [Mendeley](https://data.mendeley.com/datasets/rscbjbr9sj/2). This Mendeley dataset file is large in size due to it also having  and the computational time to run several models on home computers, we have opted to use the dataset that already includes a validation subfolder that can be found on [Kaggle](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia) for proof of concept. 

Having a high recall score for this type of neural network is important because a false negative is much more harmful than a false positive. 

## Data Preparation
After retrieving the data from the dataset it is important to classify images according to their labels. Because we are running a CNN, a supervised learning machine method, labels need to be attributed to images to help in the process of reducing loss, and increasing recall and accuracy. After this is done, each image is converted into an 1-D array, and divided by 255. The values of the grayscale pixels range from 0-255. Dividing each pixel by 255 normalizes our grayscale to values between 0-1, and also helps our CNN algorithm converge faster.

Next we needed to address our class imbalance. The first was to use a keras preprocessing function called ImageDataGenerator. ImageDataGenerator takes a batch of images used for training, applies a series of random transformation to each image in the batch (including random rotation, resizing and shearing), and replaces the original batch with the new randomly transformed batch. This effectively allows us to expand the training dataset in order to improve the performance and ability of the model to generalize.

Our second option was to divide our images classified as pneumonia between viral pneumonia, and bacterial pneumonia. We would effectively be adding another class that our model would need to classify. But we thought by splitting our pneumonia classes we may help our model by leveling out our class imbalance. Further we believed that if our model was succesfully at identify all three classes, it may aid medical practitioners to implement the appropriate medicinal therapies that would need to be implemented for the respective diagnoses of patients with viral pneumonia and bacterial pneumonia.

## Modeling

Threshhold of performance consider successful.....


## Evaluation

In evaluating our model.....

A limitation we have is access to publicaly available pediatric chest x-ray images. We would have liked to include additional data, but in our search for more, we were unable to find x-ray images from children with pneumonia and/or with nomal lungs. There are datasets available with adult lung images, however these would not be a good test set because adult and pediatric chest cavity structures are different and the model would not be able to accurately predict. A potential additional step, that is outside of the scope of this project, is to add adult to training, test, and validation sets. 


## Conclusion


## Potential Next Steps
