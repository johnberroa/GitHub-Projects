## Overview

Two implementations of perceptrons.  The Logical Perceptron creates a logical operator learning perceptron, i.e. 
it learns how to compute "and", "or", "nand", and "nor".  It returns its weights and test sample performance so that training
can be seen in the numbers.  For more understanding, it also plots the movement of the decision boundary that linearly separates
the data.  The Cluster Perceptron seperates two clusters of data.  It also prints out the weights and the results of the training on
both the training and test sets.  In theory, one can input their own cluster data and it will work out of the box.  Below are two 
pictures showing prototypical output from the algorithms:

![Logic](/Perceptrons/docs/logicperceptron.png?raw=true "Logic Perceptron")
![Cluster](/Perceptrons/docs/clusterperceptron.png?raw=true "Cluster Perceptron")
