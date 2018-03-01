# Eating-Action-Detection-using-Machine-Learning-Algorithms
Compared algorithms like SVM, Decision Tree and Deep Learning to test user actions based on metrics such as accuracy and precision.

The results displayed which algorithm performs best given the training dataset

Feature​ ​Space​ ​Selection​:

PCA was carried out on the best features obtained from the raw sensor data In the assignment 3. Using
the best features’ matrix was multiplied with the eigenvector containing highest eigenvalue to obtain
dataset in the direction where it showed most variance. Thus, the feature space was obtained. The
feature space obtained is used for the assignment 4.

Input​ ​Data​ ​(for​ ​all​ ​classification​ ​models)​:
Number of dimensions: 18 (same as from assignment 3)
Class (Eating action): 1 (last column)
Class (Non Eating action): 0 (last column)
For each user the data is picked from both the Eat_user and Noneat_user and shuffled, so that we train
and test both the classes accurately.

Labelling​ ​of​ ​the​ ​classes​:
The raw data was filtered as eating action data and non eating action data. The two
types of data were stored as two separate matrices (and csv files). Hence, it became simpler to assign
the classes to the given dataset. For SVM and decision tree classifiers, the eating action is denoted as 1
and non eating as 0. For neural networks, one-hot encoding approach is used which is the standard used
by neural network libraries in matlab. For e.g. label 0 becomes [1, 0] and label 1 becomes [0, 1]. In other
words, the index at which the value is 1 in the newly created array is used to classify the instances. This
is done because it is easier for softmax activation function to classify instances with a
confidence/probability score.

Accuracy​ ​Metrics​:
The accuracy models which are being used are as follows:

Precision​: Total Number of true positives divided by the Total number of true positives and Total
number of false positives. In other words, it is the number of positive predictions made divided by the
number of total positive classes predicted.

Recall​: Total Number of true positives divided by the Total n

Area under the curve (AUC)​: The AUC metric is used to compare two or models together which can
give an idea of the better model among all of the models.

Three different machine learning models are used to perform binary classification for the PCA
transformed dataset.
The models are run for the following variations of the dataset:
1. All 33 users and reporting the matrix containing the above classification metrics for each and
every individual user. The data for each user was shuffled during training and testing.
2. The whole dataset of 33 users is considered as one and split as training data (10 users) and
testing data (remaining 23 users) and reporting the matrix containing the above classification
metrics for each and every individual user in the testing set.

1. Decision​ ​Tree:
No​ ​of​ ​Input​ ​features 18
Classification​ ​or​ ​Regression​ ​tree Classification tree (fitctree)
Hyperparameters Default hyperparameter values
Class​ ​names 0: Non Eat and 1: Eating Action
Num​ ​of​ ​Observations 1540
Algorithm​ ​used CART algorithm (Default)
A decision tree is a tree-like representation of various data points segregated into classes based on the
attributes they possess. In Matlab, the default functionality for binary classification using decision tree is
given by the command fitctree. Here each row corresponds to an observation of the user dataset and
each column represents the predictor variable. Two classes are formed in the decision tree, namely
Eating and Non Eating. In decision trees, the model is trained and two classes are formed based on the
attributes of the training dataset. Once all the datasets are trained, testing is carried out where in each
test data is compared to the most similar class attribute, the higher the similarity in attributes, the
testing dataset is classified in that class. Thus in our case, two classes of Eating and Non Eating were
created based on the training data we provided in both the phases and results were predicted. In case of
a missing value, fitctree considers it as NaN. In decision tree, if the training dataset is very high or if it's
very similar to the testing dataset, then the classification takes place efficiently.


2.​ ​Support​ ​Vector​ ​Machines:
#​ ​of​ ​Input​ ​features 18
Type​ ​of​ ​SVM Classification SVM (fitcsvm)
Class​ ​names 0: Non Eat and 1: Eating Action
Num​ ​of​ ​Observations 1540
Kernel​ ​Used Linear (Default)
Bias 2.3472
Solver Sequential minimal optimization(SMO)
By default SVM uses linear kernel to define a model. SVM uses the classification score to classify an
instance x to be of a particular class (binary outcome in this case). The signed distance of instance x is
computed from the decision boundary ranging from -inf to +inf. A positive score means class belonging
to label 1 and negative score means it belongs to label 0.
The SVM predict method gives probabilities for different classes which is then used to classify instances
into the classes based on the highest posterior probability score.
However, by plotting the graphs of the PCA transformed dataset, it can be inferred that the data is not
linearly separable and hence linear SVM won't be able to arrive at better results. The SVM by default is a
hard margin classifier which means it can separate the two classes with a linear hyperplane. But, the soft
margin classifier has to be used because the data is messy. Hence, we can allow some points in the
training data to cross the margin. How many data points can cross the margin depends on the value of
some coefficients (slack variables). This increases the complexity of the model and can lead to
overfitting. We have to be careful here not to introduce overfitting and accordingly adjust the
coefficients. One such coefficient is called C which defines the magnitude of the margin allowed in all
dimensions. It defines the amount of violation of the margin.
● Smaller value of C means more prone to overfitting (high variance, lower bias)
● Larger value of C means less prone to overfitting (low variance, high bias)

We will want to minimise the sum of prediction error as well as the slack error in training to categorise
the data points in test data correctly.


3.​ ​Neural​ ​Networks:
#​ ​of​ ​input​ ​layer​ ​neurons​ ​(#​ ​of​ ​features) 18
Type​ ​of​ ​Neural​ ​Network Feed-forward NN
Number​ ​of​ ​hidden​ ​layers 1 (default)
Number​ ​of​ ​neurons​ ​in​ ​each​ ​hidden​ ​layer
The matrix are defined based on three algorithms: Decision Tree, SVM and Neural Networks based on
the​ ​parameters​ ​and​ ​explanation​ ​of​ ​the​ ​working​ ​of​ ​each​ ​ML​ ​algorithm​ ​is​ ​defined​ ​above.

The Result_phase1.csv file can be found in the Github Repo.

Explanation​ ​for​ ​Decision​ ​tree:
The decision tree is based on the binary classification of classes 0 and 1 where class 0 denotes Non
Eating action and class 1 represents Eating actions. We have used the inbuilt function of Matlab, fitctree
with which we are splitting the data into binary classification of two classes. As we can observe from the
above matrix, the F1 score is very high for almost all users, as we perform testing on a similar dataset on
which the training was done. It varies from mainly from 0.95 to 1, where values closer to 1 denote
better representation of the dataset by the model used. Comparing it with all the algorithms, it has a
lesser Area Under the Curve, which shows, it is not as optimal an algorithm as others.

Explanation​ ​for​ ​Support​ ​Vector​ ​Machine:
Similar to decision trees, its class 0 represents Non Eating Action and class 1 represents Eating Actions.
Based on the predictor features algorithms used by fitcsvm, it calculates probabilities for a particular
class. Based on the output of the F1 score, we can see how well is the user data classified.The F1 score
varies from 0.96 to 1, where the higher the F1 score, the better classification of the dataset is performed
by the model. Comparing it to other models, it classifies better than Decision trees, and almost similar to
Neural Networks in this case.

Explanation​ ​for​ ​Neural​ ​Network:
The number of features (18) corresponds to the number of neurons in the input layer of the neural
network. With a single hidden layer (default) and 5 neurons in each layer, the Matlab library was used
for Neural Network. Feed forward neural network was used for the processing part which was activated
by the sigmoid function. The F1 score of the Neural network, is almost similar to SVM and bit better than
Decision tree algorithm for this dataset.

Based on the final results,thus, we were able to say that the accuracy of predictions made by the Neural
Networks and SVM algorithm are the very good. We verified this by plotting the ROC curve of a single
user.
We plotted the comparison between each of this algorithms using the ROC curve where we captured
the overall efficiency based on one of the user as the sample set. The best classification accuracy is
shown by the graph which has the maximum area under the curve.

In this case each of the algorithms perform really well and it is hard to distinguish the best of them. But
analysing the results, based on every user graph and the one plotted, we can derive that Neural
Networks and SVM perform better than Decision tree in this case.

Phase​ ​2:​ ​User​ ​independent​ ​analysis
In this phase, we have used the total dataset of 33 users to train and test the given input of the feature
space for different machine learning algorithms using the following as the training and testing datasets.
For this phase:
Training​ ​on​ ​each​ ​algorithm​ ​using​ ​10​ ​users​ ​dataset.
Test​ ​using​ ​each​ ​algorithm​ ​for​ ​the​ ​remaining​ ​23​ ​users.

Using the above mentioned divisions, we ran three Machine Learning algorithms based on the above
mentioned parameters. In this case, total 23 users were tested based on the training model generated
by the 10 users dataset. We got the following matrix consisting of various metrics:
As shown in the below matrix, we have calculated Precision, Recall, F1 Score, TPR, FPR and AUC under
the ROC curve and shown in the matrix.

Each of this are given based on three algorithms: Decision Tree, SVM and Neural Networks based on the
parameters​ ​and​ ​explanation​ ​of​ ​the​ ​working​ ​of​ ​each​ ​ML​ ​algorithm​ ​is​ ​defined​ ​above.
The Result_phase2.csv file can be found in the main folder. Below is the snapshot of the same.

Explanation​ ​for​ ​Decision​ ​tree:
The decision tree is based on the binary classification of classes 0 and 1 where class 0 denotes Non
Eating action and class 1 represents Eating actions. We have used the inbuilt function of Matlab, fitctree
with which we are splitting the data into binary classification of two classes. As we can observe from the
above matrixs, the F1 score varies from the range of 0.83 to 1, where values closer to 1 denote better
representation of the dataset by the model used. Comparing it with all the algorithms, it has a lesser
Area Under the Curve, which shows, it is not as optimal an algorithm as others.

Explanation​ ​for​ ​Support​ ​Vector​ ​Machine:
Similar to decision trees, its class 0 represents Non Eating Action and class 1 represents Eating Actions.
Since the data is non separable, so the matlab function fitcsvm minimize the L1 norm problem using
various methodologies which are defined in the function directly in the Matlab. Based on the predictor
features algorithms used by fitcsvm, it calculates probabilities for a particular class. Based on the output
of the F1 score, we can see how well is the user data classified. The higher the F1 score, the better
classification of the dataset is performed by the model. Comparing it to other models, it classifies better
than Decision trees, and almost similar to Neural Networks in this case.

Explanation​ ​for​ ​Neural​ ​Network:
The number of features (18) corresponds to the number of neurons in the input layer of the neural
network. With a single hidden layer (default) and 5 neurons in each layer, the Matlab library was used
for Neural Network. Feed forward neural network was used for the processing part which was activated
by the sigmoid function. The F1 score of the Neural network, is comparatively the highest amongst all
the three algorithms for this dataset.
Based on the final results,thus, we were able to say that the accuracy of predictions made by the Neural
Networks algorithm were the best of all. We verified this by plotting the ROC curve.
We were also able to plot the comparison between each of this algorithms using the ROC curve which
captured the overall efficiency based on the training done on 10 users and testing done on remaining 23
users. The best classification accuracy is shown by the graph which has the maximum area under the
curve.
As verified using the ROC curve plotted for all the users, we see that Neural Network is the best Machine
Learning algorithm in this case, while SVM and Decision tree have a little less Area under the curve.
References​:

1. https://www.mathworks.com
