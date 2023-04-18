Assignment Writeup Links
=================
* [Homework 1 Writeup](https://github.com/gering92/EE399-Work/edit/main/README.md#homework-1-writeup)
* [Homework 2 Writeup](https://github.com/gering92/EE399-Work/edit/main/README.md#homework-2-writeup)

---

# Homework 1 Writeup
## By: Gerin George

Table of Contents
=================

* [Abstract](https://github.com/gering92/EE399-Work/edit/main/README.md#abstract)
* [Sec. I. Introduction and Overview](https://github.com/gering92/EE399-Work/edit/main/README.md#sec-i-introduction-and-overview)
* [Sec. II. Theoretical Background](https://github.com/gering92/EE399-Work/edit/main/README.md#sec-ii-theoretical-background)
* [Sec. III. Algorithm Implementation and Development](https://github.com/gering92/EE399-Work/edit/main/README.md#sec-iii-algorithm-implementation)
* [Sec. IV. Computational Results](https://github.com/gering92/EE399-Work/edit/main/README.md#sec-iv-computational-results)
* [Sec. V. Summary and Conclusion](https://github.com/gering92/EE399-Work/edit/main/README.md#sec-v-summary-and-conclusion)


### Abstract:
In this writeup, we will be analyzing a data set and fitting it to multiple different models using machine learning. We will be analyzing the error using the least squares error method. We will first find the minimum error and determine the parameters
*A, B, C,* & *D* in the equation *f(x) = A cos(Bx) + Cx + D*. After this, we will explore
the different minima that we can find if we fix two of the above parameters and sweep the other two.
Afterwards, we will split the data into training data and test data to find the least square error
with both sets of data types.

### Sec. I. Introduction and Overview
In this homework assignment, we explore various aspects of data fitting and model selection using least-squares error. We start by 
considering a dataset consisting of 31 data points, and we aim to fit the model f(x) = A cos(Bx) + Cx + D to this data with 
least-squares error. To accomplish this, we write code to determine the parameters A, B, C, and D that minimize the error.

Next, we investigate the behavior of the error landscape by fixing two of the parameters and sweeping through values of the other 
two parameters. We generate a 2D loss landscape and use pcolor to visualize the results in a grid. We examine all combinations of 
two fixed parameters and two swept parameters, and we attempt to locate the minima in the error landscape.

In the third part of the assignment, we use the first 20 data points as training data and fit three different models to the 
data: a line, a parabola, and a 19th degree polynomial. We compute the least-squares error for each of these models over the 
training points and then compute the least-square error of these models on the remaining 10 data points (the test data). We 
repeat this process, but this time we use the first 10 and last 10 data points as training data and fit the model to the test data. 
Finally, we compare the results of these two experiments.

### Sec. II. Theoretical Background
This section will cover some of the theoretical background for the concepts and techniques 
we will be using to complete this assignment.

#### Least-Squares Error
This is a common statistical technique to fit models to data. The least-squares error is defined as the 
sum of the squared differences between predicted values of the model and the actual values of the data.
Below is the Least-Squares Error equation:

![Least-Squares Error](https://user-images.githubusercontent.com/80599571/230671996-23294fdf-f84c-4431-9348-834e1d268cfd.png)

#### Parameter Optimization
Parameter optimization is the process of finding parameter values that minimize the error of the model.
This means that our model is able to better capture the dataset. We use the least-squares error method to find
the values of the parameters A, B, C, and D and minimize the error for the model f(x) = A cos(Bx) + Cx + D.
Our code iteratively adjusts the parameter values until the error is minimized. We later use optimization to 
minimize the error for different kinds of graphs, such as a line, parabola, and a 19th degree polynomial. 

#### Error Landscape
The error landscape is a visualization of the error as a function of the model parameters.
We explore the error landscape by fixing two of the parameters A, B, C, or D, and sweep
the other two parameters. We generate a 2D loss landscape using plt.pcolor(), and then 
find the minima in the error landscape by observing the color value that corresponds to
the lowest value. 


### Sec. III. Algorithm Implementation and Development
The main algorithm that was used for this homework was the least-squares error algorithm.

<img width="757" alt="image" id="Error Function for Part 1" src="https://user-images.githubusercontent.com/80599571/230986011-bd063f59-b3b5-4db4-b15f-c84ea4fc5ce7.png">


To use this algorithm, we needed to create a loss function which will return the least-squares
error. The loss function will take in a variable "c" that represents the coefficients of our model,
and variables 'x' and 'y' to represent the actual data. The loss functions will calculate the error
between our fitted curves and given data.

We also perform optimization using the 'Nelder-Mead' method. This method requires providing a
guess for the values of our parameters, and then using the optimization to store the optimized
parameter values. We then use the optimized parameters to plot the data and the fitted curve of
our model. 

Another algorithm that we use fixes two of the parameters and sweeps the other two. Since we
have 4 parameters total (A, B, C, and D), this results in 6 combinations. We do this to create 
a grid of error values which is plotted using 'pcolor'. The algorithm uses a double for loop to 
loop through a range of parameter values for two parameters while keeping the optimized values
for the other two parameters. We then plot this using 'plt.pcolor' to obtain a nice grid view.


### Sec. IV. Computational Results
#### Part 1 Results
After using the least-squares error method and running the code through the Nelder-Mead optimization,
the optimized values of the parameters were identified as:

A: 2.17175315

B: 0.90932519

C: 0.73248809

D: 31.45278269

The error that was calculated is equal to: 1.5927258504240172

![Part 1 Plot](https://user-images.githubusercontent.com/80599571/230683804-5e8ea32b-5288-4e9b-a938-02e3c0c51430.png)

#### Part 2 Results
Part 2 of the homework involved using pcolor to visualize an error grid after sweeping
2 parameters while keeping 2 parameters fixed with their optimized values. 

The plots below show the results:

![Fixing A and B, Sweeping C and D](https://user-images.githubusercontent.com/80599571/230684126-8c304a8d-141b-4cc0-88ff-b1a3b6b9ac01.png)

![Fixing A and C, Sweeping B and D](https://user-images.githubusercontent.com/80599571/230684381-4dfbd71b-d9ca-4353-9b25-7ed268212134.png)

![Fixing B and C, Sweeping A and D](https://user-images.githubusercontent.com/80599571/230684581-9cbe2e6d-9050-4b7a-b5a8-6af5140a356d.png)

![Fixing A and D, Sweeping B and C](https://user-images.githubusercontent.com/80599571/230684687-e8fb0198-8583-4b2f-8856-8b7b9ca9fc32.png)

![Fixing B and D, Sweeping A and C](https://user-images.githubusercontent.com/80599571/230684765-899306f1-f4bf-440b-9bed-24bfc630854a.png)

![Fixing C and D, Sweeping A and B](https://user-images.githubusercontent.com/80599571/230684830-f8927fdd-4cd5-4a83-8f61-7e65c63a34db.png)

#### Part 3 Results
Part 3 of the homework was the calculation of error for various types of curves being fit to the data.
The given data was split up into training data and test data. There were 31 data points, so for part 3, the first twenty were treated
as training data and the last 11 are treated as test data. Using this, I defined least square error functions for a line, parabola, and
a 19th degree polynomial. Using the function for the least square error, I was able to print out the values for each.
The results of these error calculations are printed below:

Least Square Error for ***Line (Training Data):*** **2.242749386808538**

Least Square Error for ***Line (Test Data):*** **3.36363873604787**

Least Square Error for ***Parabola (Training Data):*** **2.1255393482773766**

Least Square Error for ***Parabola (Test Data):*** **8.713651781874919**

Least Square Error for ***19th Deg Polynomial (Training Data):*** **0.028351503968806435**

Least Square Error for ***19th Deg Polynomial (Test Data):*** **28617752784.428474**

The errors were all generally good enough, with the line having the best amount of error. The 19th degree polynomial 
had very low error for the training data, but it had an insanely high error for test data. I am assuming this is due to
overfitting occurring due to having a smaller data set to work with.

#### Part 4 Results
Part 4 of the homework was very similar to part 3, but involved changing which indices of the dataset would be used for training and test.
The training data now captures the first 10, and the last 10 of the data set. The test data captures the middle 11. The rest of the 
process between Part 3 and Part 4 were the same.
The results of the error calculations are printed below:

Least Square Error for ***Line (Training Data):*** **2.8684634748880655**

Least Square Error for ***Line (Test Data):*** **22.197891223912386**

Least Square Error for ***Parabola (Training Data):*** **2.8680459400504987**

Least Square Error for ***Parabola (Test Data):*** **22.571695465713965**

Least Square Error for ***19th Deg Polynomial (Training Data):*** **0.692679558738857**

Least Square Error for ***19th Deg Polynomial (Test Data):*** **154987332439.0542**

The 19th degree polynomial test data was similar very high as in part 3. This indicates to me a severe overfitting as a result of not having enough data points. Error increased for the line and parabola as well. 

### Sec. V. Summary and Conclusions

This work has illustrated how to fit a curve to a set of points from a dataset. The least squares error method was used to effectively be able to produce a curve that could attempt fit data. We observed the visual effect of minimas by fixing two parameters and sweeping the other two. We also observed the differences between a line, parabola, and a 19 degree polynomial when trying to fit data. We observed the effects of overfitting, as the error skyrocketed and became a major outlier. Overall, the process of fitting curves to data in machine learning is an important tool that allows us to make predictions and gain insights from complex datasets. By using the least squares error method, we can effectively determine the best curve that fits the data and make accurate predictions based on this curve. However, it is important to be cautious of overfitting, as this can lead to erroneous predictions and inaccurate insights. The process of fitting curves to data is an important aspect of machine learning that should be used with care and attention to detail in order to achieve accurate and meaningful results.

---
# Homework 2 Writeup

## By: Gerin George

Table of Contents
=================

* [Abstract](https://github.com/gering92/EE399-Work/edit/main/README.md#abstract-1)
* [Sec. I. Introduction and Overview](https://github.com/gering92/EE399-Work/edit/main/README.md#sec-i-introduction-and-overview)
* [Sec. II. Theoretical Background](https://github.com/gering92/EE399-Work/edit/main/README.md#sec-ii-theoretical-background)
* [Sec. III. Algorithm Implementation and Development](https://github.com/gering92/EE399-Work/edit/main/README.md#sec-iii-algorithm-implementation)
* [Sec. IV. Computational Results](https://github.com/gering92/EE399-Work/edit/main/README.md#sec-iv-computational-results)
* [Sec. V. Summary and Conclusion](https://github.com/gering92/EE399-Work/edit/main/README.md#sec-v-summary-and-conclusion)

### Abstract
This homework assignment involves working with a dataset of 39 faces, each with approximately 65 lighting scenes. We'll be using Python and various mathematical techniques to analyze the data and gain insights into its structure.

We'll start by computing a correlation matrix between the first 100 images in the dataset and plotting it using pcolor. From this matrix, we'll identify the two most highly correlated and uncorrelated images and plot their corresponding faces. We'll also repeat this process for a smaller subset of images.

Next, we'll use eigenvector decomposition and SVD to identify the first six principal component directions of the dataset. We'll compare the first eigenvector with the first SVD mode and compute the norm of the difference between their absolute values. Finally, we'll also calculate the percentage of variance captured by each of the first six SVD modes and plot them.

### Sec. I. Introduction and Overview
In part A, we are asked to plot a correlation matrix of the matrix X, which contains a dataset of 39 faces with 65 lighting scenes giving us 2414 faces to analyze. We first create a 100 x 100 correlation matrix C by computing the dot products of the first 100 images in the matrix X, which has columns of individual images. We plot the correlation matrix using pcolor to see the correlation patterns.

In part B, we are asked to use the correlation matrix that we made to then find the two most correlated images and the two most uncorrelated images and to plot those 4 images. We find the most correlated images using ```np.unravel_index(np.argmax(C[utri]), C.shape)```. This code finds the maximum value index in the matrix C to find the position of the images with the most correlation. Similarly, ```np.unravel_index(np.argmin(C[utri]), C.shape)``` is used to find the minimum value, with the only difference being that we use np.argmin instead of np.argmax. 

In part C, we repeat part a but with a 10 x 10 correlation matrix instead. 

In part D, we find the 6 first six eigenvectors with the largest magnitude eigenvalue. 

Part E involved doing single value decomposition (SVD) on the matrix X containing all the images, and finding the six principal component directions. 

In part F, we are comparing the first eigenvector v<sub>1</sub> that we calculated from part D with the SVD mode u<sub>2</sub> from part E. We then compute the norm of difference of their absolute values.

Part G, the final part, involves computing the percentage of variance from each of the first 6 SVD modes, and plotting the first 6 SVD modes. 
 
### Sec. II. Theoretical Background

In regards to machine learning, the correlation matrix that we created by computing the dot product between the matrix X and the transpose of the matrix X allows us to see how closely two images are related. By observing the value of their dot product, we know how similar two images can be. Using the correlation matrix, we can then find the most and least correlated images

Eigenvectors are a type of vector that represent the directions in which a linear transformation preserves its shape. In image processing, eigenvectors can be used to represent the dominant patterns or features present in a set of images. Eigenvectors with the largest magnitude eigenvalues are known as principal components, and can be used to project the images into a lower dimensional space. 

We are also asked to perform an SVD on the matrix X and find the first six principal component directions. Singular Value Decomposition (SVD) is a powerful technique used in linear algebra and data anlysis. It is used to decompose a matrix into its constituent parts in order to perform a range of operations such as dimensionality reduction, data comporession, and feature extraction. The key idea of SVD, is to represent a matrix as a product of three matrixes: U, &Sigma;, and V<sup>T</sup>. U and V are orthogonal matrixes, and &Sigma; is a diagonal matrix with non-negative real numbers on the diagonal. The SVD of a matrix X is expressed as: 
A = U * &Sigma; * V<sup>T</sup>

In terms of dimensionality reduction, by keeping only the most important singular values and corresponding columsn of U and V, we can reduce the dimensionality of the original matrix while still preserving most of its variance. Overall, SVD is a powerful technique with important applications in machine learning. 

### Sec. III. Algorithm Implementation and Development
Initialization of sample data into the matrix X: 
```
results=loadmat('yalefaces.mat')
X=results['X']
```

### Sec. IV. Computational Results

### Sec. V. Summary and Conclusion
