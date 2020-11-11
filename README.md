# Pneumonia Detection

Our goal for this project is to classify and train a convoluted neural network (CNN) model to predict if an x-ray image is of lungs with pneumonia or normal.

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

### Business Understanding: 

According to [Mayo Clinic](https://www.mayoclinic.org/diseases-conditions/pneumonia/symptoms-causes/syc-20354204), "Pneumonia is an infection that inflames the air sacs in one or both lungs. The air sacs may fill with fluid or pus (purulent material), causing cough with phlegm or pus, fever, chills, and difficulty breathing. A variety of organisms, including bacteria, viruses and fungi, can cause pneumonia." Particularly for children, pneumonia can be very dangerous. 

"Pneumonia is the single largest cause of death in children worldwide. Every year, it kills an estimated 1.4 million children under the age of five years, accounting for 18% of all deaths of children under five years old worldwide." These statistics from the [World Health Organization](https://www.who.int/maternal_child_adolescent/news_events/news/2011/pneumonia/en/) show the importance of an accurate and timely diagnosis of the infection. 

There is often a [backlog](https://schwanerinjury.com/delayed-mri-ct-scan-or-x-ray-can-death-injury/) of x-ray results due to a shortage of radiologists. Some hospitals have tried to address this problem by outsourcing x-rays to other countries in different time zones to have more coverage. 

This neural network does not mean to replace any radiologists' inspection of an x-ray. If a child is getting a chest x-ray it is likely already considered a priority case.  Using this neural netowork could help doctors and hospital administrators think about ways to help their patient prioritization protocols, how to possibly expedite the process for the pedriatric patients with suspected pneumonia, and whether or not it is helpful to them. 

The stakeholders here are the doctors treating pediatric patients with suspected pneumonia. 

### Data Understanding:

The dataset is comprised of greyscale x-ray images of pediatric patients. Within these, 1,583 show normal lungs, and 4,273 show lungs with pneumonia. The original dataset comes from Kermany et al. on [Mendeley](https://data.mendeley.com/datasets/rscbjbr9sj/2). Due to the large size of the Mendeley dataset and the computational time to run several models on home computers, we have opted to use the paired down dataset that can be found on [Kaggle](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia) for proof of concept. 

Having a high recall score for this type of neural network is important because a false negative is much more harmful than a false positive. 

## Data Preparation

Dealing with class imbalance

## Modeling

Threshhold of performance consider successful:

## Evaluation


## PHASE 2????

## Conclusion


## Potential Next Steps
