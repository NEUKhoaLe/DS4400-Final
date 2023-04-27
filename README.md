# DS4400-Final

## Introduction:

The problem arises during COVID times, and also inspired by police interrogation footage. During COVID times, 
as well as recent research of artificial therapist models, where learning facial expression is especially 
important. What we are trying to solve is the process of classifying facial expression for use during therapy 
sessions and police interrogation footage. By using Machine Learning to classify facial expression, we can speed up the 
time needed to prescreen patients and interrogation suspects. Especially for artificial intelligence therapy models, 
the models can check for the patient's emotions and give the therapist more insight into the patient's psyche, speeding up 
the time between pre-screening and diagnosis. The problem is interesting because it tackles image classification, 
which is one of the core concepts of Deep Learning.

The Approach I propose to tackle the problem is to use varying strength of CNN models to test for performance when 
classifying these images. CNN makes sense for this because we are classifying images. They will work well, as they can
 give us the important features of an image while reducing input feature space. In terms of reference, I refer to simple 
 CNNs that are used for classifying MNIST problems, as well as kaggle submission. Due to the time constraints, my strongest model has similar 
 parameters to a sample model, with dropout rates being the same. The limitation is that the more complex the classification task is, the 
 longer it will need to be trained, and accuracy will be lost. There is a trial and error process in choosing the CNN configurations.
 
 ## Setup:
 For the dataset, it was taken from Kaggle. This dataset has 7 classes of facial 
 expression to classify. In total, there are 28,709 training data, and 3589 testing data. The class "disgust" has the least amount of data, 
 which leads to poor performance when classifying disgust faces. The images are in grayscale, and sized at 48x48 pixels. The images are helpful in that 
 it automatically centers on the subject faces. 
 
 The experimental setup is 4 CNN models, with each model increasing in their layers. The first two models are 
 baseline models that have two CNN layers. The difference between the two are that one is doing Max pooling and the other is doing average pooling. 
 The next two models have more convoluting and pooling layers, and that the main difference bewteen the two is that the strongest CNN model have batch 
 normalization and random dropouts, which helps with preventing overfitting. The main goal of the project is to do the classification problem, and 
 test for performance between average and max pooling, and the different configurations that you can have for CNNs. The parameters are 20 epochs, and 32 
 batch sizes. We will be running the experiments in Jupyter notebook using Pytorch for creating the models. 
 
 ## Results:
 The main results is that the strongest model performs the best as expected. The most interesting finding is that the base line model 
 performs acceptably, with a 56% test accuracy. However, because we only trained with 20 epochs, and that the loss graphs still show no 
 sign of plateauing, we could have gone for more epochs. Another interesting finding is that there are certain classes that the model performs 
 well on, such as classifying happy and sad. However, the models have trouble classifying disgust and angry, both due to the similarity and the lack of 
 training data. 
 
 For our strongest models, we decided to do dropouts between every layer, including the classifying layers. We also do batch normalization between each 
 convolution layer, to speed up processing time. However, because of the complexity of the model, each epoch still took 5 minutes of training time. If we could 
 afford the training time, we would have gone for 40 or 60 epochs to maximize performance. I would also like to increase more hidden layers in the 
 feed forward layer, because we might not have enough in terms of finding the relations between the filters and layers that we convolved. 
 
 ## Discussion
 
 The results that we obtained were satisfactory, especially for common emotions such as happiness and sadness. However, it is unsatisfactory for 
 emotions that are tougher to classify, like disgust and anger. I think a potential problem is the lack of longer training time, as well as the lack of 
 an extra hidden layer. With longer training time, we might achieve better accuracy.
 
 ## Conclusion: 
 
 In this project, I made 4 differenct CNN models to classify facial expressions into 7 different categories. Between each CNN models, I compared their 
 performance and implemented them differently and uniquely. The strongest CNN model has similar structure to recommended settings, such as batch normalization and 
 perceptron dropouts, which leads to better accuracy but much longer training time. I also included a sample script from a website that runs an OpenCV function to detact faces 
 in any images, and output them. In the future, we can use this function to pass through extracted faces into our strongest CNN model to automatically classify any 
 images.
