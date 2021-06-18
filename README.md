# BERT-Ghazali

Al-Ghazali's authorship attribution
===================================

Purpose:
--------

The product is a suitable functional learning-machine intended to be used by researchers in order to evaluate the authorship attribution of Al-Ghazali’s manuscripts. the program allows control over the parameters and methods used by the algorithm.

Program structure:
------------------

![](https://www.dropbox.com/s/zjihw5i7ihw7n0y/struct.png?dl=1)

First, at the home screen the user can choose between starting a new research process or review older research.

-       New research:  define parameter for the new research and run it.
-       Review old research: reload previously saved results and parameters        of the reseach. In this point there are 2 furthers options:

-   -   Show results: this option loads the results view and let the user remember the results recieved under the given parameters.
    -   Re-run: there is a small measure of randomness in the algorithm. it means that the same parameters may yield slightly different results. The re-run option allows the user to try the same parameter over and over again to examine how the results converge into a solid answer.

Training status view - keeps track of the training and classification process. Including partial accuracies over the training process and remaining times for each part of the classification.

Results view - composed of three main parts:

1.  Heat map - visual demonstratation of the classifications and matching-percentage for the classified books in the test set in each iteration.
2.  Cluster centroids - there are 2 clusters: Written by Al-Ghazali and Not written by Al-Ghazali. different parameters results different classification values. Cluster centroids is visual exhibit of those clusters. The further the centroids from each other means solid results.
3.  Final results - Table of 2 columns: Book name and Classification. 

Last is the Save view where the user can save the results and parameters of the current research for later review.

How to use:
-----------

There are two main aspects need a deep review when defining a new research.

General Configurations:

-   Niter - the number of iteration of classification. the classification process will repeat Niter times and finally will average the results.
-   Accuracy threshold - a fraction in range [0,1]. New CNN is trained every iteration and verified against validation set. The CNN will proceed to the classification phase only if the validation accuracy is equal or better than the accuracy threshold.
-   Silhouette threshold - a fraction in range [0,1]. After the Niter CNN's classifies the test set, a silhouette value for the clusters is calculated. The user may demand a minimum silhouette value. Lower silhouette value will alert the user with red bold label and will not allow saving the current result.
-   F1 and F - parameters to handle the imbalanced data sets problem. the default values are the optimal we found. to learn more click [here](https://machinelearningmastery.com/random-oversampling-and-undersampling-for-imbalanced-classification/).
-   BERT input length - BERT is the algorithm used to produce word embeddings. BERT has limitation of input length it may process. Different length will require producing new set of word embeddings for all the data sets.
-   Text division method - Since BERT has input length limitation, [the text needs](https://textfancy.com) to be broken into parts. 
    -   Fixed size division - every |BERT input length| tokens is a fraction of the text.
    -   Bottom up - build whole sentences. Avoiding breaking sentences in the middle, which result some embeddings out of context.

**NOTICE: The program produces and saves previously used embeddings. if the embeddings under the given configurations (Bert input length and Text division method, combined) do not exist it will produce them, which may TAKE UP TO 4 HOURS based on your hardware.** If the embeddings configuration already used before - the program will use it. 

CNN Configurations:

-   Kernels - number of kernel in the CNN. Number of kernels must be matching to the number of arguments given in 1-D convolutional kernels field.
-   1-D conv kernels - The sizes of the convolutional kernels. Number of given values must be matching to the number in Kernels field.
-   Strides - The size of stride of the convolutional kernel over the embedding matrix.
-   Batch size - The number of samples processed before the model is updated.
-   Learning rate - a hyperparameter that controls how much to change the model in response to the estimated error each time the model weights are updated.
-   Dropout - removes units during training, reducing the capacity of the network.
-   Epochs -  The number of complete passes through the training dataset. 
-   Activation function - Relu or Sigmoid.

Notes:
------

Contacts:

Ron Raifer: [ronraifer@gmail.com](mailto:ronraifer@gmail.com)

Asaf ben shabat: [asafbenshabat@gmail.com](mailto:asafbenshabat@gmail.com) 
