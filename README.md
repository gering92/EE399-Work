# *Assignment Writeup Links*

=================
* [Homework 1 Writeup](https://github.com/gering92/EE399-Work/blob/main/README.md#homework-1-writeup)
* [Homework 2 Writeup](https://github.com/gering92/EE399-Work/blob/main/README.md#homework-2-writeup)
* [Homework 3 Writeup](https://github.com/gering92/EE399-Work/blob/main/README.md#homework-3-writeup)
* [Homework 4 Writeup](https://github.com/gering92/EE399-Work/blob/main/README.md#homework-4-writeup)
* [Homework 5 Writeup](https://github.com/gering92/EE399-Work/blob/main/README.md#homework-5-writeup)
* [Homework 6 Writeup](https://github.com/gering92/EE399-Work/blob/main/README.md#homework-6-writeup)
---

# Homework 1 Writeup
## By: Gerin George

Table of Contents
=================

* [Abstract](https://github.com/gering92/EE399-Work/blob/main/README.md#abstract)
* [Sec. I. Introduction and Overview](https://github.com/gering92/EE399-Work/blob/main/README.md#sec-i-introduction-and-overview)
* [Sec. II. Theoretical Background](https://github.com/gering92/EE399-Work/blob/main/README.md#sec-ii-theoretical-background)
* [Sec. III. Algorithm Implementation and Development](https://github.com/gering92/EE399-Work/blob/main/README.md#sec-iii-algorithm-implementation-and-development)
* [Sec. IV. Computational Results](https://github.com/gering92/EE399-Work/blob/main/README.md#sec-iv-computational-results)
* [Sec. V. Summary and Conclusions](https://github.com/gering92/EE399-Work/blob/main/README.md#sec-v-summary-and-conclusions)


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

* [Abstract](https://github.com/gering92/EE399-Work/blob/main/README.md#abstract-1)
* [Sec. I. Introduction and Overview](https://github.com/gering92/EE399-Work/blob/main/README.md#sec-i-introduction-and-overview-1)
* [Sec. II. Theoretical Background](https://github.com/gering92/EE399-Work/blob/main/README.md#sec-ii-theoretical-background-1)
* [Sec. III. Algorithm Implementation and Development](https://github.com/gering92/EE399-Work/blob/main/README.md#sec-iii-algorithm-implementation-and-development-1)
* [Sec. IV. Computational Results](https://github.com/gering92/EE399-Work/blob/main/README.md#sec-iv-computational-results-1)
* [Sec. V. Summary and Conclusion](https://github.com/gering92/EE399-Work/blob/main/README.md#sec-v-summary-and-conclusion)

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

#### Part A: Computing a 100 x 100 Correlation Matrix C:
The Python code below isolates the first 100 images, and then uses the np.dot function to calculate the dot product between X_100 and X_100<sup>T</sup>
```
# Compute the correlation matrix using dot product
X_100 = X[:, :100]
C = np.dot(X_100.T, X_100)
```
We then plot C using pcolor. 

#### Part B: Find the two most highly correllated images and the two lowest correllated images

The python code below is used to find the indices of the upper triangle of the matrix C to improve efficiency.
```
utri = np.triu_indices(C.shape[0], k=1)
```

The code below uses np.unravel_index to find the indices of the maximum values. 
```
max_idx = np.unravel_index(np.argmax(C[utri]), C.shape)
img1 = X_100[:, max_idx[0]]
img2 = X_100[:, max_idx[1]]
```
A similar method using np.argmin is used to find the indices of the minimum values. 

These images are then plotted. 

#### Part C: Computing a 10 x 10 Correlation Matrix:
The python code below isolates the first 100 images and uses the np.dot function to create a correlation matrix. 
```
C = np.dot(X[:, [0, 312, 511, 4, 2399, 112, 1023, 86, 313, 2004]].T, X[:, [0, 312, 511, 4, 2399, 112, 1023, 86, 313, 2004]])
```
After computing the new 10 x 10 correlation matrix, we plot it to observe the new correlation values. 

#### Part D: Creating New Matrix Y and Finding the First Six Eigenvectors with the Largest Magnitude Eigenvalue 
The code below calculates the Matrix Y, which is the dot product of X and X<sup>T</sup>. We then find the eigenvalues and eigenvectors using np.linalg.eigh(), which returns both eigenvalues and eigenvectors. 
```
Y = np.dot(X, X.T)
eigenvals, eigenvecs = np.linalg.eigh(Y)
```
The code below is to sort the Eigenvalues in descending order so we can find the eigenvectors with the largest eigenvalue: 
```
idx = np.argsort(eigenvals)[::-1]
```
The code below is to find the first 6 eigenvectors. 
```
top_eigenvecs = eigenvecs[:, idx[:6]]
```
We then have to normalize the eigenvectors in order to ensure that they have a unit length. This makes it easier to interpret the eigenvectors and see them as directions in space that are most relevant to distinguishing facial images. 
```
norms = np.linalg.norm(top_eigenvecs, axis=0)
normalized_eigenvecs = top_eigenvecs / norms
```

We then print out the 6 eigenvectors.

#### Part E: Singular Value Decomposition (SVD) and Finding the First Six Principal Component Directions

To perform singular value decomposition, we can use the np.linalg.svd() function, which returns the three matrices U, V, and &Sigma.
```
U, s, V = np.linalg.svd(X)
```

The principal component directions are the columns of V<sup>T</sup>, and we can find the first 6 using the code below.
```
principal_component_directions = V[:6,:].T
```

#### Part F: Comparing the First Eigenvector from Part D and the First SVD Mode from Part E and Calculating Norm of Difference of Absolute Value

The code below is used to capture the first Eigenvector and the first SVD mode. 
```
eigenvector_1 = normalized_eigenvecs[:, 0]
svd_mode_1 = U[:, 0]
```
We then find the norm of difference of their absolute values using this code: 
```
norm_diff = np.linalg.norm(np.abs(eigenvector_1) - np.abs(svd_mode_1))
```
Afterwards, we can just print out norm_diff.

#### Part G: Computing Percentage of Variance from First 6 SVD Modes and Plotting the First 6 SVD Modes

The code below calculates the sum of squares of the projections onto each of the six SVD modes. This is necessary to determine the variance captured by the 6 SVD modes.
```
ss_projections = np.sum((X.T @ U[:, :6])**2, axis=0)
```
We can find the total variance of the origianl data by doing a similar calculation:
```
total_variance = np.sum(X**2)
```

We can then find the percent of variance captured by each of the first 6 SVD modes by dividing ss_projections by total_variance. 
```
variance_percentages = ss_projections / total_variance * 100
```

### Sec. IV. Computational Results

#### Part A: 
The correlation matrix that was created from the code is seen below:

<img width="480" alt="image" src="https://user-images.githubusercontent.com/80599571/232940032-13bae3f4-13ed-4a0a-955c-1b10c6c7e7ce.png">

We can see a criss cross patttern in the correlation matrix. The yellow coloring means images that share high correlation, and dark blue represents low correlation. 

#### Part B: 
The two most correlated and two most uncorrelated images can be seen below:

<img width="480" alt="Image of most correlated and least correlated images" src="https://user-images.githubusercontent.com/80599571/232940864-50126652-7e46-4767-a5e3-7c1696da3c46.png">

From here, we can see that the most uncorrelated images are the ones with a stark difference in lighting, while the most correlated images share similar lighting features. 

#### Part C: 
The result of part C is similar to part A, but with a 10 x 10 matrix instead using specific indices. The resulting correlation matrix is shown below:

<img width="480" alt="image" src="https://user-images.githubusercontent.com/80599571/232949278-5d4ad62c-ee94-44ee-aeb2-e62366764a16.png">

#### Part D: 
The first six eigenvectors with the largest eigvenvalues are seen in the image below:

<img width="600" alt="image" src="https://user-images.githubusercontent.com/80599571/232943633-e26e7dff-0a54-4b50-b329-272fb0154613.png">


#### Part E: 
The first six principal component directions are seen in the image below:

<img width="555" alt="image" src="https://user-images.githubusercontent.com/80599571/232947400-139be950-6b61-439b-aa91-660a4a127548.png">

#### Part F:
The calculated norm of the absolute difference between the first eigenvector and the first SVD mode is printed below: 

```
6.125883123259166e-16
```

This shows that the difference between the eigenvector and the SVD mode is very small. 

#### Part G:
The variances in the SVD Modes are printed below: 
```
Variances: 
SVD Mode 1: 72.9276
SVD Mode 2: 15.2818
SVD Mode 3: 2.5667
SVD Mode 4: 1.8775
SVD Mode 5: 0.6393
SVD Mode 6: 0.5924
```

The plots of the variance in SVD modes are in the image below:
<img width="791" alt="image" src="https://user-images.githubusercontent.com/80599571/232948874-cdc9e95e-94e8-49dd-88ed-1b71e9bf3d94.png">


### Sec. V. Summary and Conclusion

In this homework, we explored more about the techniques used in machine learning. This includes the use of dot products to find correlation matrices, the process of SVD and how it is useful to reduce a matrix into three matrices which captures patterns and relationships in data. It can be used to project data into a lower dimensional space while retaining much of the original structure. We also explore the importance of eigenvectors and eigenvalues in this assignment. 

We first applied dot products to this assignment by computing a 100 x 100 correlation matrix of the first 100 images in the dataset. We then saw how well this correlation matrix worked by capturing the most correlated and most uncorrelated image pairs from the data. 

We then began to observe how eigenvectors and SVD are related by finding the first six eigenvectors with the largest magnitude eigenvalues of the dot product of X and X<sup>T</sup>. Singular Value Decomposition was performed on the matrix X after this to find the first six principal component directions. 

To observe the similar results obtained from these methods, we calculated the norm of difference of their absolute values. The value we obtained was 6.125883123259166e-16, illustrating how principal component directions and the eigenvectors with the largest magnitude eigenvalues are similar. 

The last step involved finding the percentage of variance of the first 6 SVD modes. This resulted in 6 variance values, with the first SVD mode capturing the highest variance at 72.93% and the 6th SVD mode capturing the lowest variance at 0.59%. 

The analysis performed in this homework on the set of images from Yalefaces helps us understand the approach that is used to find similarities in images. With correlation matrices, eigenvectors, and SVDs, we are able to find similarities in images and use machine learning to recognize faces.

---
# Homework 3 Writeup

## By: Gerin George

Table of Contents
=================

* [Abstract](https://github.com/gering92/EE399-Work/blob/main/README.md#abstract-2)
* [Sec. I. Introduction and Overview](https://github.com/gering92/EE399-Work/blob/main/README.md#sec-i-introduction-and-overview-2)
* [Sec. II. Theoretical Background](https://github.com/gering92/EE399-Work/blob/main/README.md#sec-ii-theoretical-background-2)
* [Sec. III. Algorithm Implementation and Development](https://github.com/gering92/EE399-Work/blob/main/README.md#sec-iii-algorithm-implementation-and-development-2)
* [Sec. IV. Computational Results](https://github.com/gering92/EE399-Work/blob/main/README.md#sec-iv-computational-results-2)
* [Sec. V. Summary and Conclusion](https://github.com/gering92/EE399-Work/blob/main/README.md#sec-v-summary-and-conclusion-1)


### Abstract

This homework assingment involves working with the MNIST dataset. This dataset contains handwritten numbers from 0-9. The analysis involves performing an SVD analysis of the images, projecting them onto PCA space, and building classifiers to identify individual digits. The rank of the digit space is determined by analyzing the singular value spectrum, and the interpretation of the U, Σ, and V matrices is discussed. A 3D plot is created to show the projection of the data onto three selected V-modes. Linear classifiers (LDA), SVMs, and decision tree classifiers are built to classify different pairs and groups of digits, and their performance is evaluated on both the training and test sets. The difficulty of separating different pairs of digits is also discussed. The assignment requires creating visualizations to aid in the analysis and understanding of the results.

### Sec. I. Introduction and Overview

The MNIST dataset is a well-known benchmark dataset in machine learning, consisting of 70,000 grayscale images of handwritten digits from 0 to 9. In this assignment, we will perform an analysis of the MNIST dataset using singular value decomposition (SVD) and principal component analysis (PCA), as well as building classifiers to identify individual digits in the training set using linear discriminant analysis (LDA), support vector machines (SVM), and decision trees.

### Sec. II. Theoretical Background

In the first part, we will reshape each image into a column vector and perform an SVD analysis to determine the singular value spectrum and the necessary number of modes for good image reconstruction. We will also interpret the U, Σ, and V matrices, and project the data onto three selected V-modes to create a 3D plot colored by digit label.

In the second part, we will build classifiers to identify individual digits in the training set using LDA, SVM, and decision trees. We will first pick two digits and try to build a linear classifier that can reasonably identify/classify them, then pick three digits and build a linear classifier to identify these three. We will also identify the two digits in the dataset that are most difficult to separate and quantify the accuracy of the separation with LDA on the test data. Similarly, we will identify the two digits in the dataset that are easiest to separate and quantify the accuracy of the separation with LDA on the test data. Finally, we will compare the performance between LDA, SVM, and decision trees on the hardest and easiest pair of digits to separate.

Throughout the assignment, we will use visualizations to help us better understand and interpret the data and results.

### Sec. III. Algorithm Implementation and Development

Initialization of the sample MNIST data. We divide by 255.0 to normalize the data, and then transpose to make each column a different image. 

```python
# Load the MNIST data
mnist = fetch_openml('mnist_784')
X = np.array(mnist.data / 255.0) # Normalizes data and puts data in matrix X

X = X.T
```

#### Question 1: SVD Analysis of the Digit Images

The following code performs SVD analysis and gathers the output matrices in the variables U, S, and Vt:
```python
# Perform SVD on the images
U, S, Vt = np.linalg.svd(X, full_matrices=False)
```

#### Question 2: Singular Value Spectrum Plot and Rank r of Digit Space

The following code will plot the S matrix, which has the singular value spectrum. 
```python
#Plot the singular values
fig = plt.figure(figsize=(15, 8))  # set figure size
plt.plot(S)
plt.xlabel('Singular value index')
plt.ylabel('Singular value')
plt.xticks(range(0, len(S), 20))  # set x-ticks at intervals of 25
plt.yticks(range(0, 2000, 100))  # set y-ticks at intervals of 100
```

We then use the following code to determine the rank r, and place it on our plot:
```python
# Determine the rank r
threshold = 0.1 * S[0]
r = np.sum(S > threshold)

plt.plot(r, S[r], 'ro', label=f'Rank {r}')  # add a red dot at the rank r value and label it

plt.title('MNIST Digit Images Singular Value Spectrum')
plt.legend()
plt.show()

print('The Singular Values look like an exponential decay.')
print('Rank of digit space:', r)
```

#### Question 3: Interpretation of the U, &Sigma;, and V Matrices:

The U, &Sigma;, and V matrices represent the factorization of a matrix that undergoes singular value decomposition. 

The U matrix represents the left singular vectors, and span the column space of the matrix A that undergoes SVD.

The V matrix represents the right singular vectors, and span the row space of the matrix A that undergoes SVD. 

The &Sigma; matrix represents the singular values, which are the square roots of the eigenvalues of the matrix A. They indicate the importance of each singular vector in the data, and are sorted in descending order along the diagonal of S. 

#### Question 4: 3D Plot of Digit Data Projected onto V-Modes 2, 3, and 5

The following code is used to select three modes (2, 3, and 5) from V. We then use matrix multiplication to project it. 

```python
# Project the data onto the three selected V-modes
V_selected = Vt[:, [1, 2, 4]]
projected_data = (X.T @ V_selected)
```

The code below extracts the digit labels into the matrix y:

```python
y = np.array(mnist.target)
```

The code below plots a 3D scatterplot. np.unique(y) is used to grab every unique value in y, which are digits 0-9. A mask is used from these unique values to create the plot. 

```python
unique_vals = np.unique(y)
num_vals = len(unique_vals)
print(num_vals)

# Plot the scatter plot
fig = plt.figure(figsize=(20,20))
ax = fig.add_subplot(111, projection='3d')


colors = ['red', 'green', 'blue', 'cyan', 'magenta', 'yellow', 'black', 'purple', 'gray', 'orange']
for i, val in enumerate(unique_vals):
    mask = np.where(y == val)[0]
    ax.scatter3D(projected_data[mask, 0], projected_data[mask, 1], projected_data[mask, 2], color=colors[i], label=str(val))


# Set the axis labels
ax.set_xlabel('V_2')
ax.set_ylabel('V_3')
ax.set_zlabel('V_5')
ax.set_title('Projection of MNIST onto V-modes 2, 3, and 5')

# Add the legend
ax.legend(loc='upper left', title='Digit Labels')

plt.show()
```

#### Two Digit Linear Classification

The two digits that were selcted to be used for linear classification are 3 and 8.  

```python
# Select two digits to classify (e.g., 3 and 8)
digit1 = '3'
digit2 = '8'
```

We use these to isolate X and y for occurences where they appear:

```python
# Use the indices to select the samples and labels
X_selected = X[:, (y == digit1) | (y == digit2)]
y_selected = y[(y == digit1) | (y == digit2)]
```

The data is split into training and test data by using the train_test_split function:
```python
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_selected.T, y_selected, test_size=0.2, random_state=42)
```

We then perform an LDA, obtain the score, and then print out the accuracy:

```python
# Perform LDA with n_components=1
lda = LDA(n_components=1)
lda.fit(X_train, y_train)

# Get the predicted labels for the test set
training_score = lda.score(X_train, y_train)

test_score = lda.score(X_test, y_test)


# Accuracy of training
print("Training Accuracy: {:.3f}%".format(training_score*100))

# Accuracy of Test data
print("Test Accuracy: {:.3f}%".format(test_score*100))
```

#### Three Digit LDA:

The three digit classification is very similar to the two digit classification. 

Three digits are selected:

```python
digit1 = '3'
digit2 = '8'
digit3 = '9'
```

Then we select the right columns: 

```python
# Use the indices to select the samples and labels
X_selected = X[:, (y == digit1) | (y == digit2) | (y == digit3)]
y_selected = y[(y == digit1) | (y == digit2) | (y == digit3)]
```

The rest of the code involves splitting into test and training data, then performing LDA, similar to the last part.

```python
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_selected.T, y_selected, test_size=0.2, random_state=42)

# Perform LDA with n_components=1
lda = LDA(n_components=1)
lda.fit(X_train, y_train)

# Get the predicted labels for the test set
training_score = lda.score(X_train, y_train)

test_score = lda.score(X_test, y_test)


# Accuracy of training
print("Training Accuracy: {:.3f}%".format(training_score*100))

# Accuracy of Test data
print("Test Accuracy: {:.3f}%".format(test_score*100))
```

#### Most Difficult Digits and Easiest Digits to Separate:

To determine the most difficult digits to separate, the combinations function from itertools is used. We create an array of all the digits, and then run. combinations through that to find all the digit combinations. 

```python
from itertools import combinations

# Define the digits to classify
digits = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

# Get all possible combinations of two digits
digit_pairs = combinations(digits, 2)
```

We also keep track of max and min accuracies for test and training with the following block of code: 

```python
# Initialize variables to store max and min test accuracy and their corresponding digit pairs
max_test_accuracy = 0
max_test_accuracy_digits = None
min_test_accuracy = 100
min_test_accuracy_digits = None
```

The for loop below does the same LDA analysis that we did above, but with every single digit pair from the combinations function output passed in as a digit pair. We split into training and test data using train_test_split(), and then perform an LDA. We obtain the score by using lda.score, and then update the max and min accuracy variables accordingly using an if block check.

```python
for digit_pair in digit_pairs:
    digit1, digit2 = digit_pair
    
    # Select the samples and labels for the current digit pair
    X_selected = X[:, np.isin(y, [digit1, digit2])]
    y_selected = y[np.isin(y, [digit1, digit2])]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_selected.T, y_selected, test_size=0.2, random_state=42)

    # Perform LDA with n_components=1
    lda = LDA(n_components=1)
    lda.fit(X_train, y_train)

    training_score = lda.score(X_train, y_train)
    # Get the predicted labels for the test set
    test_score = lda.score(X_test, y_test)
    
    print("Digits {} vs {}: Training Accuracy: {:.3f}%".format(digit1, digit2, training_score*100))

    # Print the test accuracy for the current digit pair
    print("Digits {} vs {}: Test Accuracy: {:.3f}%".format(digit1, digit2, test_score*100))
    print('\n')
    
    # Update the max and min test accuracy and their corresponding digit pairs
    if test_score > max_test_accuracy:
        max_test_accuracy = test_score
        max_test_accuracy_digits = digit_pair
    if test_score < min_test_accuracy:
        min_test_accuracy = test_score
        min_test_accuracy_digits = digit_pair
```

The max and min output is printed like this:

```python
# Print the max and min test accuracy and their corresponding digit pairs
print("Maximum Test Accuracy: {:.3f}% (Digits {} vs {})".format(max_test_accuracy*100, max_test_accuracy_digits[0], max_test_accuracy_digits[1]))
print("Minimum Test Accuracy: {:.3f}% (Digits {} vs {})".format(min_test_accuracy*100, min_test_accuracy_digits[0], min_test_accuracy_digits[1]))
```

#### Support Vector Machine (SVM) and Decision Tree Comparison:

On the full data set, we use the train_test_split() function. 

```python
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

Then, using the training data, we fit an SVM, Decision Tree, and LDA model:

```python
# Fit an SVM classifier
svm_clf = svm.SVC()
svm_clf.fit(X_train, y_train)

# Fit a decision tree classifier
dt_clf = DecisionTreeClassifier()
dt_clf.fit(X_train, y_train)

# Fit an LDA model
lda = LDA(n_components=1)
lda.fit(X_train, y_train)
```

We then use predict functions for each to make a prediction based on the training data on the test data:

```python
# Make predictions on the test set
svm_preds = svm_clf.predict(X_test)
dt_preds = dt_clf.predict(X_test)
```

We calculate the accuracies using the code below:

```python
# Calculate and print the accuracies
svm_acc = accuracy_score(y_test, svm_preds)
dt_acc = accuracy_score(y_test, dt_preds)
lda_acc = lda.score(X_test, y_test)
```

The output is printed like this:

```python
print("SVM Accuracy: {:.3f}%".format(svm_acc*100))
print("Decision Tree Accuracy: {:.3f}%".format(dt_acc*100))
print("LDA Accuracy: {:.3f}%".format(lda_acc*100))
```

#### LDA, SVM and Decision Tree Comparison on Hardest and Easiest Pair of Digits

An SVM, Decision Tree, and LDA model were trained using the training data for the hardest pair of digits, 3 and 5:

```python
# Select the samples and labels for the hardest digit pair (3, 5)
X_selected = X[np.logical_or(y == '3', y == '5')]
y_selected = y[np.logical_or(y == '3', y == '5')]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_selected, y_selected, test_size=0.2, random_state=42)
```

The models were fit using the following code:

```python
# Fit an SVM classifier
svm_clf = svm.SVC()
svm_clf.fit(X_train, y_train)

# Fit a decision tree classifier
dt_clf = DecisionTreeClassifier()
dt_clf.fit(X_train, y_train)

# Fit an LDA model
lda = LDA(n_components=1)
lda.fit(X_train, y_train)
```

Predictions were made, and then the accuracy was taken like this, which is similar to what was used in the previous section:

```python
# Make predictions on the test set
svm_preds = svm_clf.predict(X_test)
dt_preds = dt_clf.predict(X_test)
lda_preds = lda.predict(X_test)

# Calculate and print the accuracies
svm_acc = accuracy_score(y_test, svm_preds)
dt_acc = accuracy_score(y_test, dt_preds)
lda_acc = lda.score(X_test, y_test)
```

A similar analysis to the above was done for the easiest digit pair, 6 and 8.

```python
# Select the samples and labels for the hardest digit pair (6, 8)
X_selected = X[np.logical_or(y == '6', y == '8')]
y_selected = y[np.logical_or(y == '6', y == '8')]
```

The outputs were printed like this:

```python
print("Performance on the Hardest Pair (3, 5):")
print("SVM Accuracy: {:.3f}%".format(svm_acc*100))
print("Decision Tree Accuracy: {:.3f}%".format(dt_acc*100))
print("LDA Accuracy: {:.3f}%".format(lda_acc*100))
print("\n")
```

```python
print("Performance on the Easiest Pair (6, 7):")
print("SVM Accuracy: {:.3f}%".format(svm_acc*100))
print("Decision Tree Accuracy: {:.3f}%".format(dt_acc*100))
print("LDA Accuracy: {:.3f}%".format(lda_acc*100))
```


### Sec. IV. Computational Results

#### Question 1:

Question 1 was just to perform an SVD, so it had no results beyond the output matrices. 

#### Question 2:

The output plot of the singular value spectrum and the rank of the digit space is seen below:

<img width="503" alt="image" src="https://user-images.githubusercontent.com/80599571/233768514-d944c04a-e8f4-4050-b2a2-db5ec5bb5291.png">


#### Question 3:

In the context of image analysis, the matrix U is a set of basis images that capture the main patterns in the digit images (The original images in the MNIST dataset). A column of U corresponds to a basis image. The matrix V is a set of basis coefficients that determine the contribution of each basis image to a particular digit image. Each row of V corresponds to coefficients of the basis images for a particular digit image. The singular values in the matrix S can be thought of as the importance of each basis image. The larger the singular value in S, the more important the corresponding basis image.  

#### Question 4: 

The 3D scatter plot below represents the digit labels projected onto V-modes 2, 3 and 5. 

<img width="499" alt="image" src="https://user-images.githubusercontent.com/80599571/233768841-23a4e53c-efc6-4482-a563-3a7caa76f774.png">

#### LDA on Two Digits:

Linear Discriminant Analysis on digits 3 and 8 resulted in the following accuracies for training and test data:

```
Training Accuracy: 96.724%
Test Accuracy: 96.528%
```

#### LDA on Three Digits:

Linear Discriminant Analysis on digits 3, 8, and 9 resulted in the following accuracies for training and test data:

```
Training Accuracy: 95.723%
Test Accuracy: 94.719%
```

#### Two Hardest and Easiest Digits to Separate:

The text below represent the training and test accuracies for all combinations of digits. We then print out the max and min values for the test data to determine the hardest and easiest pair of digits to separate. 

```
Digits 0 vs 1: Training Accuracy: 99.476%
Digits 0 vs 1: Test Accuracy: 99.526%


Digits 0 vs 2: Training Accuracy: 98.812%
Digits 0 vs 2: Test Accuracy: 98.633%


Digits 0 vs 3: Training Accuracy: 99.448%
Digits 0 vs 3: Test Accuracy: 98.825%


Digits 0 vs 4: Training Accuracy: 99.681%
Digits 0 vs 4: Test Accuracy: 99.381%


Digits 0 vs 5: Training Accuracy: 98.865%
Digits 0 vs 5: Test Accuracy: 98.487%


Digits 0 vs 6: Training Accuracy: 99.274%
Digits 0 vs 6: Test Accuracy: 98.621%


Digits 0 vs 7: Training Accuracy: 99.674%
Digits 0 vs 7: Test Accuracy: 99.401%


Digits 0 vs 8: Training Accuracy: 98.907%
Digits 0 vs 8: Test Accuracy: 98.689%


Digits 0 vs 9: Training Accuracy: 99.504%
Digits 0 vs 9: Test Accuracy: 98.738%


Digits 1 vs 2: Training Accuracy: 98.613%
Digits 1 vs 2: Test Accuracy: 98.151%


Digits 1 vs 3: Training Accuracy: 99.034%
Digits 1 vs 3: Test Accuracy: 98.169%


Digits 1 vs 4: Training Accuracy: 99.694%
Digits 1 vs 4: Test Accuracy: 99.422%


Digits 1 vs 5: Training Accuracy: 99.154%
Digits 1 vs 5: Test Accuracy: 98.802%


Digits 1 vs 6: Training Accuracy: 99.670%
Digits 1 vs 6: Test Accuracy: 99.085%


Digits 1 vs 7: Training Accuracy: 99.300%
Digits 1 vs 7: Test Accuracy: 98.978%


Digits 1 vs 8: Training Accuracy: 97.517%
Digits 1 vs 8: Test Accuracy: 96.226%


Digits 1 vs 9: Training Accuracy: 99.671%
Digits 1 vs 9: Test Accuracy: 99.259%


Digits 2 vs 3: Training Accuracy: 97.532%
Digits 2 vs 3: Test Accuracy: 96.781%


Digits 2 vs 4: Training Accuracy: 98.769%
Digits 2 vs 4: Test Accuracy: 97.901%


Digits 2 vs 5: Training Accuracy: 98.280%
Digits 2 vs 5: Test Accuracy: 97.182%


Digits 2 vs 6: Training Accuracy: 98.494%
Digits 2 vs 6: Test Accuracy: 98.234%


Digits 2 vs 7: Training Accuracy: 98.722%
Digits 2 vs 7: Test Accuracy: 98.355%


Digits 2 vs 8: Training Accuracy: 97.277%
Digits 2 vs 8: Test Accuracy: 96.634%


Digits 2 vs 9: Training Accuracy: 98.969%
Digits 2 vs 9: Test Accuracy: 98.566%


Digits 3 vs 4: Training Accuracy: 99.436%
Digits 3 vs 4: Test Accuracy: 99.105%


Digits 3 vs 5: Training Accuracy: 96.302%
Digits 3 vs 5: Test Accuracy: 95.020%


Digits 3 vs 6: Training Accuracy: 99.438%
Digits 3 vs 6: Test Accuracy: 99.001%


Digits 3 vs 7: Training Accuracy: 98.727%
Digits 3 vs 7: Test Accuracy: 98.268%


Digits 3 vs 8: Training Accuracy: 96.724%
Digits 3 vs 8: Test Accuracy: 96.528%


Digits 3 vs 9: Training Accuracy: 98.182%
Digits 3 vs 9: Test Accuracy: 97.447%


Digits 4 vs 5: Training Accuracy: 99.182%
Digits 4 vs 5: Test Accuracy: 98.744%


Digits 4 vs 6: Training Accuracy: 99.307%
Digits 4 vs 6: Test Accuracy: 99.015%


Digits 4 vs 7: Training Accuracy: 98.689%
Digits 4 vs 7: Test Accuracy: 98.407%


Digits 4 vs 8: Training Accuracy: 99.487%
Digits 4 vs 8: Test Accuracy: 98.718%


Digits 4 vs 9: Training Accuracy: 97.016%
Digits 4 vs 9: Test Accuracy: 95.865%


Digits 5 vs 6: Training Accuracy: 97.773%
Digits 5 vs 6: Test Accuracy: 97.650%


Digits 5 vs 7: Training Accuracy: 99.394%
Digits 5 vs 7: Test Accuracy: 98.971%


Digits 5 vs 8: Training Accuracy: 96.527%
Digits 5 vs 8: Test Accuracy: 96.347%


Digits 5 vs 9: Training Accuracy: 98.794%
Digits 5 vs 9: Test Accuracy: 98.493%


Digits 6 vs 7: Training Accuracy: 99.876%
Digits 6 vs 7: Test Accuracy: 99.612%


Digits 6 vs 8: Training Accuracy: 98.558%
Digits 6 vs 8: Test Accuracy: 98.577%


Digits 6 vs 9: Training Accuracy: 99.774%
Digits 6 vs 9: Test Accuracy: 99.494%


Digits 7 vs 8: Training Accuracy: 99.026%
Digits 7 vs 8: Test Accuracy: 98.725%


Digits 7 vs 9: Training Accuracy: 96.482%
Digits 7 vs 9: Test Accuracy: 95.405%


Digits 8 vs 9: Training Accuracy: 98.050%
Digits 8 vs 9: Test Accuracy: 97.461%


Maximum Test Accuracy: 99.612% (Digits 6 vs 7)
Minimum Test Accuracy: 95.020% (Digits 3 vs 5)
```

From the data, we see that the model had the easiest time separating digits 6 and 7. It had the hardest time distinguishing digits 3 and 5. 

#### SVM and Decision Tree Accuracy for all Ten Digits

The accuracies of SVM and Decision Tree classifiers for all ten digits compared to the LDA accuracy is printed below:

```
SVM Accuracy: 97.643%
Decision Tree Accuracy: 87.114%
LDA Accuracy: 86.771%
```

SVM was actually the most accurate, as LDA had an accuracy of 86.771% when all ten digits were introduced. 

#### SVM, Decision Tree Classifier, and LDA Accuracy for Hardest and Easiest Digit Pairs

On the hardest and easiest pairs of digits to separate, the models performed as printed below:

```
Performance on the Hardest Pair (3, 5):
SVM Accuracy: 99.257%
Decision Tree Accuracy: 96.507%
LDA Accuracy: 95.020%


Performance on the Easiest Pair (6, 7):
SVM Accuracy: 100.000%
Decision Tree Accuracy: 99.012%
LDA Accuracy: 99.612%
```

SVM was in general the most accurate, as it got 100% on the easiest pair, (6, 7), and 99.257% on the hardest pair for LDA (3, 5). 

### Sec. V. Summary and Conclusion

This homework explored Singular Value Decomposition further, and the matrices that are involved. We dug a little deeper into the linear algebra that makes this all possible. We analyzed the MNIST data set, and explored a linear discriminant analysis model on the data. We analyzed how well it was able to identify two, then three digits, and then found the hardest and easiest digits for the LDA to separate. 

Afterwards, we built an SVM (Support Vector Machine) and Decision Tree Classifier model, and asked them to separate all ten digits. We compared it to the LDA. We then compared all three models performance again on the hardest and easiest pair of digits to separate. 

We found that in general, the SVM was most capable of identifying digits most accurately. 

---
# Homework 4 Writeup
## By: Gerin George



* [Abstract](https://github.com/gering92/EE399-Work/blob/main/README.md#abstract-3)
* [Sec. I. Introduction and Overview](https://github.com/gering92/EE399-Work/blob/main/README.md#sec-i-introduction-and-overview-3)
* [Sec. II. Theoretical Background](https://github.com/gering92/EE399-Work/blob/main/README.md#sec-ii-theoretical-background-3)
* [Sec. III. Algorithm Implementation and Development](https://github.com/gering92/EE399-Work/blob/main/README.md#sec-iii-algorithm-implementation-and-development-3)
* [Sec. IV. Computational Results](https://github.com/gering92/EE399-Work/blob/main/README.md#sec-iv-computational-results-3)
* [Sec. V. Summary and Conclusion](https://github.com/gering92/EE399-Work/blob/main/README.md#sec-v-summary-and-conclusion-2)


### Abstract


In this homework assignment, we explore the application of feedforward neural networks for fitting and analyzing datasets. We will compare the performance of these neural networks with the models developed in previous assignments (HW1 and HW3) using temperature data and the MNIST dataset. The main focus will be on implementing and evaluating a three-layer feedforward neural network.

The first part of the assignment involves fitting the temperature data to a feedforward neural network and evaluating the least-square error for various training and testing splits. The performance of the neural network will be compared with the models created in HW1. In the second part, we will work with the MNIST dataset by computing the first 20 PCA modes of the digit images and building a feedforward neural network for digit classification. The neural network's performance will be compared against other classification techniques such as LSTM, SVM (support vector machines), and decision tree classifiers. 

### Sec. I. Introduction and Overview

In recent years, neural networks have emerged as a powerful tool for modeling complex data structures and solving various machine learning tasks. Among the different types of neural networks, feedforward neural networks have gained popularity due to their simplicity and effectiveness. In this assignment, we aim to explore the application of feedforward neural networks in fitting datasets and comparing their performance with previously developed models.

The assignment consists of two main parts. In the first part, we revisit the temperature data from homework one and fit it to a three-layer feedforward neural network. The performance of the neural network will be evaluated using the least-square error metric for different training and testing data splits. This analysis will provide insights into the neural network's ability to generalize and adapt to new data. A comparison will be made between the neural network and the models developed in HW1 to assess their relative performance.

In the second part of the assignment, we shift our focus to the widely-used MNIST dataset. We begin by computing the first 20 PCA modes of the digit images, which serve as a reduced representation of the data, capturing most of the variance in the dataset. Next, we build a feedforward neural network to classify the handwritten digits and compare its performance against other popular classification techniques such as LSTM, SVM, and decision tree classifiers. This comparison will help us understand the strengths and weaknesses of different classifiers in the context of digit recognition.

### Sec. II. Theoretical Background

The theoretical background behind this assignment involves a few key concepts and algorithms in the field of machine learning and neural networks. These include feedforward neural networks, backpropagation, principal component analysis (PCA), and various classification techniques.

Feed Forward Neural Networks (FFNNs): FFNNs are a type of artificial neural network where the connections between nodes do not form a cycle. The information flows in one direction, from the input layer through hidden layers (if any) to the output layer. Each layer consists of nodes, also known as neurons or units, that apply an activation function to the weighted sum of their inputs. FFNNs are widely used for supervised learning tasks, such as regression and classification.

Backpropagation: Backpropagation is a supervised learning algorithm used to train neural networks. It is a form of supervised learning that adjusts the weights of the network to minimize the error between the predicted output and the actual output (ground truth). This is achieved through gradient descent optimization, which computes the gradient of the error with respect to each weight and updates the weights accordingly. The process is repeated iteratively until the error converges to a minimum value.

Principal Component Analysis (PCA): PCA is a dimensionality reduction technique used to transform data into a lower-dimensional space while preserving most of the variance in the dataset. PCA finds a set of orthogonal axes, known as principal components, that capture the maximum variance in the data. By projecting the original data onto these principal components, a reduced representation of the data is obtained, which can be used for various machine learning tasks, including classification and visualization.

Classification Techniques: In this assignment, we compare the performance of feedforward neural networks against other popular classification techniques, such as Long Short-Term Memory (LSTM) networks, Support Vector Machines (SVM), and Decision Trees. These techniques have their own strengths and weaknesses, depending on the problem domain and the structure of the data:

 - LSTM networks are a type of recurrent neural network (RNN) that can learn long-term dependencies in sequential data. They are particularly useful for tasks involving time series or natural language processing.

 - SVM is a supervised learning method that aims to find the optimal hyperplane that separates the classes in the feature space. SVMs are effective in high-dimensional spaces and can handle nonlinear classification using kernel functions.

 - Decision Trees are hierarchical structures that recursively split the data based on feature values, aiming to maximize the purity of the resulting subsets. They are easy to interpret and can handle both numerical and categorical features.

### Sec. III. Algorithm Implementation and Development

#### Question 1: FFNN on 2D Temperature Data from HW1

To implement the feed forward neural network, a custom class called Net was created using the code below:

```python
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(1, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
```

This FFNN has three layers, with the first layer having an input of 1 and an output of 64, the second having an input of 64 and an output of 32, and the third layer having an input of 32 and an output of 1. The forward method is used to apply the ReLU activation function to feed the input tensor x forward through each layer until it reaches the final output layer. 

The 2D temperature data is loaded and separated into the first training and test configuration below.

```python
X = np.arange(0, 31)
Y = np.array([30, 35, 33, 32, 34, 37, 39, 38, 36, 36, 37, 39, 42, 45, 45, 41,
              40, 39, 42, 44, 47, 49, 50, 49, 46, 48, 50, 53, 55, 54, 53])

train_X = torch.tensor(X[:20], dtype=torch.float32).view(-1, 1)
train_Y = torch.tensor(Y[:20], dtype=torch.float32).view(-1, 1)

test_X = torch.tensor(X[20:], dtype=torch.float32).view(-1, 1)
test_Y = torch.tensor(Y[20:], dtype=torch.float32).view(-1, 1)
```

In the first configuration, the first twenty points are taken as training, and the last ten are taken as test.

We create a Net() object net, and give it the criterion of mean square error for the loss function. We also use Stochastic Gradient Descent (SGD) as the optimizer.

```python
net = Net()
criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=0.001)
```

The learning rate in the code above is set to 0.001, which is the step size that is taken when updating the parameters by the optimizer. 

The code below is training the FFNN on the training data over 10,000 epochs:

```python
num_epochs = 10000
for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = net(train_X)
    loss = criterion(outputs, train_Y)
    loss.backward()
    optimizer.step()

    if (epoch+1) % 100 == 0:
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))
```

The code below is used to find the accuracy of the neural net on the data by feeding the predicted values and the training data to the criterion function:

```python
with torch.no_grad():
    train_pred = net(train_X)
    train_error = criterion(train_pred, train_Y)
    print("Least-square error on training data: {:.4f}".format(train_error.item()))

    test_pred = net(test_X)
    test_error = criterion(test_pred, test_Y)
    print("Least-square error on test data: {:.4f}".format(test_error.item()))
```

Similarly to the first configuration, the second configuration does practically the same thing. The only difference is in the data points we use as training and test. IN the second configuration, we use the first 10 and the last 10 as training data, and the middle 10 as test data. 

```python
train_X = torch.tensor(np.concatenate((X[:10], X[-10:])), dtype=torch.float32).view(-1, 1)
train_Y = torch.tensor(np.concatenate((Y[:10], Y[-10:])), dtype=torch.float32).view(-1, 1)

test_X = torch.tensor(X[10:20], dtype=torch.float32).view(-1, 1)
test_Y = torch.tensor(Y[10:20], dtype=torch.float32).view(-1, 1)
```

The rest of the code works the exact same way, where we instantiate a neural net model and feed it the training data to train it, and then scoring its accuracy using the criterion function. We are also still using the MSE as the loss calculator, and are also using Stochastic Gradient Descent. 

```python
net = Net()
optimizer = optim.SGD(net.parameters(), lr=0.001)

for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = net(train_X)
    loss = criterion(outputs, train_Y)
    loss.backward()  # Added parentheses here
    optimizer.step()

    if (epoch+1) % 100 == 0:
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))

with torch.no_grad():
    train_pred = net(train_X)
    train_error = criterion(train_pred, train_Y)
    print("Least-square error on training data (first 10 and last 10): {:.4f}".format(train_error.item()))

    test_pred = net(test_X)
    test_error = criterion(test_pred, test_Y)
    print("Least-square error on test data (10 held out middle data points): {:.4f}".format(test_error.item()))
```

#### Question 2: FFNN on MNIST Data from HW3

Similarly to what we did in HW3, we are now working with the MNIST dataset and trying to fit a neural network to the data. First, we have to load the MNIST data by using the code below:

```python
# Load the MNIST data
X, y = mnist.data / 255.0, mnist.target
y = y.astype(np.int)
```

After this, we compute the first twenty principal component analysis modes of the digits in the MNIST dataset. 

```python
# Compute the first 20 PCA modes of the digit images
pca = PCA(n_components=20)
pca.fit(X)

# Get the first 20 PCA modes (principal components)
pca_modes = pca.components_
```

After this, we can build the FFNN we will be using for this modeling using a custom class. 

```python
# Define the neural network architecture
class FFNN(nn.Module):
    def __init__(self):
        super(FFNN, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
```

Using the train_test_split() function, we are able to split the data into training and test data. 

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

We then convert the data into tensors using the following method:

```python
# Convert the data to numpy arrays
X_train_np = X_train.to_numpy()
X_test_np = X_test.to_numpy()

y_train_np = y_train.to_numpy()
y_test_np = y_test.to_numpy()

# Convert the data to tensors
X_train_torch = torch.tensor(X_train_np, dtype=torch.float32)
y_train_torch = torch.tensor(y_train_np, dtype=torch.long)
X_test_torch = torch.tensor(X_test_np, dtype=torch.float32)
y_test_torch = torch.tensor(y_test_np, dtype=torch.long)
```

Due to the size of the MNSIT dataset, we create a Data Loader to make the stochastic gradient descent both faster and less computationally intense. 

```python
# Create DataLoader for the training data
train_data = TensorDataset(X_train_torch, y_train_torch)
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
```

We instantiate the FFNN and set up the loss function and optimizer using the code below. We still use a learning rate of 0.001:

```python
# Instantiate the neural network, set up the loss function and optimizer
ffnn = FFNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(ffnn.parameters(), lr=0.001)
```

The FFNN is trained over 10 epochs using the following code:

```python
# Train the neural network
num_epochs = 10
for epoch in range(num_epochs):
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = ffnn(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
```

The FFNN's accuracy is determine using the accuracy_score() function and the code below:

```python
# Evaluate the neural network
with torch.no_grad():
    y_pred_ffnn = ffnn(X_test_torch).argmax(dim=1).numpy()
    ffnn_accuracy = accuracy_score(y_test, y_pred_ffnn)
    print("Feed-forward neural network accuracy: {:.4f}".format(ffnn_accuracy))
```

The next part of this assignment has us compare the accuracies of the FFNN to different classifing methods. 

The first one we will be exploring is the LSTM, which stands for Long Short-Term Memory. The LSTM is built using the code below:

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTMClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        # Initialize hidden state and cell state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        
        # LSTM forward pass
        out, _ = self.lstm(x, (h0, c0))
        
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out
```

The code below is used to establish different model parameters. 

```python
# LSTM model parameters
hidden_size = 128
num_layers = 2
num_classes = 10
```

The hidden_size parameter is meant to represent the number of neurons in each LSTM layer. A larger hidden size value means that the model can learn more complex patterns in the data. Increasing it too much can lead to overfitting though, so it is good to make it a good balance between large and small. 

The num_layers parameter is the number of layers in the LSTM model. Multiple layers can help the model learn more complex representations of the data, but adding too many layers can lead to more parameters and more chance of overfitting occurring. 

num_classes parameter is the number of output classes for the classification problem. Since we have 10 digits, 0-9, we have 10 possible outputs in the context of the MNIST dataset. 

The code below is used to reshape the input data for the kind of input that is expected of the LSTM model. The input size is equal to 28 since each image in the NMIST dataset is 28 x 28 pixels. The LSTM model will process one row of 28 pixels at a time. the sequence length corresponds to the number of time steps the LSTM model will process, and it matches the number of rows in each 28 x 28 pixel image. 

We create new X_train_LSTM and X_test_LSTM input devices that are the reshaped version of the training data in a suitable format for the LSTM model. The shape of the variable X_train_LSTM and X_test_LSTM will be (batch_size, sequence_length, input_size).

An instance of the LSTMClassifier class is created, and it moves the model to the CPU using the device variable. 

```python
# Reshaping input data for the LSTM model
input_size = 28
sequence_length = 28
X_train_LSTM = X_train_torch.view(-1, sequence_length, input_size)
X_test_LSTM = X_test_torch.view(-1, sequence_length, input_size)

lstm_model = LSTMClassifier(input_size, hidden_size, num_layers, num_classes).to(device)
```

We create the loss function and the optimizer for the LSTM using the code below. The loss function that is used is the Cross Entropy Loss loss function from PyTorch. Cross Entropy loss works by measuring the difference between two probability distributions, the predicted distribution and the true distribution. 

The optimizer is configured for a learning rate of 0.001 as its step size. 

```python
# Set the loss function and optimizer for LSTM
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(lstm_model.parameters(), lr=0.001)
```

The code below is used to train the LSTM model:

```python
# Train the LSTM model
num_epochs = 10
batch_size = 100

for epoch in range(num_epochs):
    for i in range(0, len(X_train_LSTM), batch_size):
        inputs = X_train_LSTM[i:i + batch_size].to(device)
        labels = y_train_torch[i:i + batch_size].to(device)

        optimizer.zero_grad()
        outputs = lstm_model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
```

10 epochs are being used for the training. 

The code below is used to test the LSTM model and print out the accuracy:

```python
# Test the LSTM model
with torch.no_grad():
    correct = 0
    total = 0
    for i in range(0, len(X_test_LSTM), batch_size):
        inputs = X_test_LSTM[i:i + batch_size].to(device)
        labels = y_test_torch[i:i + batch_size].to(device)
        output = lstm_model(inputs)
        _, predicted = torch.max(output.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(f'LSTM Accuracy: {correct / total}')
```

We then compare how SVM and Decision Trees handle the same MNIST dataset by training them on the dataset and printing the accuracies. 

The code below is used to fit an SVM model to the data and find its accuracy:

```python
# SVM fit and test   
svm = SVC(kernel='rbf', random_state=42)
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)

svm_accuracy = accuracy_score(y_test, y_pred_svm)
print("SVM accuracy: {:.4f}".format(svm_accuracy))
```

The code below is used to fit a decision tree to the data and find its accuracy:

```python
# Decision tree fit and test
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)

dt_accuracy = accuracy_score(y_test, y_pred_dt)
print("Decision tree accuracy: {:.4f}".format(dt_accuracy))
```

### Sec. IV. Computational Results

#### Question 1: 2D Temperature Data

In the first question, we are seeing how the FFNN model handles the 2D temperature data from the first homework assignment. The output below is the results of the FFNN on the first configuration, where the first 20 points are taken as training data and the last 10 are taken as test data.

```
Epoch [100/10000], Loss: 81.0442
Epoch [200/10000], Loss: 22.3072
Epoch [300/10000], Loss: 65.6368
Epoch [400/10000], Loss: 27.9052
Epoch [500/10000], Loss: 14.5961
Epoch [600/10000], Loss: 9.4846
Epoch [700/10000], Loss: 7.2426
Epoch [800/10000], Loss: 6.0671
Epoch [900/10000], Loss: 5.3451
Epoch [1000/10000], Loss: 4.8606
Epoch [1100/10000], Loss: 4.5989
Epoch [1200/10000], Loss: 4.4452
Epoch [1300/10000], Loss: 4.3482
Epoch [1400/10000], Loss: 4.2930
Epoch [1500/10000], Loss: 4.2551
Epoch [1600/10000], Loss: 4.2248
Epoch [1700/10000], Loss: 4.1965
Epoch [1800/10000], Loss: 4.1705
Epoch [1900/10000], Loss: 4.1459
Epoch [2000/10000], Loss: 4.1227
Epoch [2100/10000], Loss: 4.1013
Epoch [2200/10000], Loss: 4.0815
Epoch [2300/10000], Loss: 4.0633
Epoch [2400/10000], Loss: 4.0465
Epoch [2500/10000], Loss: 4.0311
Epoch [2600/10000], Loss: 4.0172
Epoch [2700/10000], Loss: 4.0044
Epoch [2800/10000], Loss: 3.9923
Epoch [2900/10000], Loss: 3.9808
Epoch [3000/10000], Loss: 3.9699
Epoch [3100/10000], Loss: 3.9595
Epoch [3200/10000], Loss: 3.9496
Epoch [3300/10000], Loss: 3.9403
Epoch [3400/10000], Loss: 3.9314
Epoch [3500/10000], Loss: 3.9231
Epoch [3600/10000], Loss: 3.9152
Epoch [3700/10000], Loss: 3.9078
Epoch [3800/10000], Loss: 3.9008
Epoch [3900/10000], Loss: 3.8942
Epoch [4000/10000], Loss: 3.8881
Epoch [4100/10000], Loss: 3.8823
Epoch [4200/10000], Loss: 3.8770
Epoch [4300/10000], Loss: 3.8719
Epoch [4400/10000], Loss: 3.8673
Epoch [4500/10000], Loss: 3.8629
Epoch [4600/10000], Loss: 3.8589
Epoch [4700/10000], Loss: 3.8552
Epoch [4800/10000], Loss: 3.8518
Epoch [4900/10000], Loss: 3.8489
Epoch [5000/10000], Loss: 3.8478
Epoch [5100/10000], Loss: 3.8664
Epoch [5200/10000], Loss: 4.0083
Epoch [5300/10000], Loss: 3.9868
Epoch [5400/10000], Loss: 3.9253
Epoch [5500/10000], Loss: 3.9217
Epoch [5600/10000], Loss: 3.9328
Epoch [5700/10000], Loss: 3.9333
Epoch [5800/10000], Loss: 3.9245
Epoch [5900/10000], Loss: 3.9194
Epoch [6000/10000], Loss: 3.9160
Epoch [6100/10000], Loss: 3.9124
Epoch [6200/10000], Loss: 3.9085
Epoch [6300/10000], Loss: 3.9050
Epoch [6400/10000], Loss: 3.9008
Epoch [6500/10000], Loss: 3.8974
Epoch [6600/10000], Loss: 3.8944
Epoch [6700/10000], Loss: 3.8910
Epoch [6800/10000], Loss: 3.8874
Epoch [6900/10000], Loss: 3.8837
Epoch [7000/10000], Loss: 3.8811
Epoch [7100/10000], Loss: 3.8779
Epoch [7200/10000], Loss: 3.8748
Epoch [7300/10000], Loss: 3.8731
Epoch [7400/10000], Loss: 3.8705
Epoch [7500/10000], Loss: 3.8680
Epoch [7600/10000], Loss: 3.8656
Epoch [7700/10000], Loss: 3.8635
Epoch [7800/10000], Loss: 3.8605
Epoch [7900/10000], Loss: 3.8586
Epoch [8000/10000], Loss: 3.8568
Epoch [8100/10000], Loss: 3.8552
Epoch [8200/10000], Loss: 3.8539
Epoch [8300/10000], Loss: 3.8524
Epoch [8400/10000], Loss: 3.8506
Epoch [8500/10000], Loss: 3.8492
Epoch [8600/10000], Loss: 3.8468
Epoch [8700/10000], Loss: 3.8456
Epoch [8800/10000], Loss: 3.8450
Epoch [8900/10000], Loss: 3.8436
Epoch [9000/10000], Loss: 3.8435
Epoch [9100/10000], Loss: 3.8419
Epoch [9200/10000], Loss: 3.8410
Epoch [9300/10000], Loss: 3.8403
Epoch [9400/10000], Loss: 3.8391
Epoch [9500/10000], Loss: 3.8385
Epoch [9600/10000], Loss: 3.8374
Epoch [9700/10000], Loss: 3.8356
Epoch [9800/10000], Loss: 3.8359
Epoch [9900/10000], Loss: 3.8344
Epoch [10000/10000], Loss: 3.8349
Least-square error on training data: 3.8342
Least-square error on test data: 20.5051
```

We see from the output that the loss started out relatively high, but settled around 3.8 after enough Epochs had passed. The LSE on the training data was much lower than on the test data, which indicates some overfitting had occurred. 

The output of the second configuration is shown below:

```
Epoch [100/10000], Loss: 1338.1541
Epoch [200/10000], Loss: 61.1255
Epoch [300/10000], Loss: 57.7992
Epoch [400/10000], Loss: 56.6340
Epoch [500/10000], Loss: 56.9259
Epoch [600/10000], Loss: 56.4704
Epoch [700/10000], Loss: 56.4068
Epoch [800/10000], Loss: 56.3196
Epoch [900/10000], Loss: 55.5626
Epoch [1000/10000], Loss: 56.4000
Epoch [1100/10000], Loss: 56.4000
Epoch [1200/10000], Loss: 56.4000
Epoch [1300/10000], Loss: 56.4000
Epoch [1400/10000], Loss: 56.4000
Epoch [1500/10000], Loss: 56.4000
Epoch [1600/10000], Loss: 56.4000
Epoch [1700/10000], Loss: 56.4000
Epoch [1800/10000], Loss: 56.4000
Epoch [1900/10000], Loss: 56.4000
Epoch [2000/10000], Loss: 56.4000
Epoch [2100/10000], Loss: 56.4000
Epoch [2200/10000], Loss: 56.4000
Epoch [2300/10000], Loss: 56.4000
Epoch [2400/10000], Loss: 56.4000
Epoch [2500/10000], Loss: 56.4000
Epoch [2600/10000], Loss: 56.4000
Epoch [2700/10000], Loss: 56.4000
Epoch [2800/10000], Loss: 56.4000
Epoch [2900/10000], Loss: 56.4000
Epoch [3000/10000], Loss: 56.4000
Epoch [3100/10000], Loss: 56.4000
Epoch [3200/10000], Loss: 56.4000
Epoch [3300/10000], Loss: 56.4000
Epoch [3400/10000], Loss: 56.4000
Epoch [3500/10000], Loss: 56.4000
Epoch [3600/10000], Loss: 56.4000
Epoch [3700/10000], Loss: 56.4000
Epoch [3800/10000], Loss: 56.4000
Epoch [3900/10000], Loss: 56.4000
Epoch [4000/10000], Loss: 56.4000
Epoch [4100/10000], Loss: 56.4000
Epoch [4200/10000], Loss: 56.4000
Epoch [4300/10000], Loss: 56.4000
Epoch [4400/10000], Loss: 56.4000
Epoch [4500/10000], Loss: 56.4000
Epoch [4600/10000], Loss: 56.4000
Epoch [4700/10000], Loss: 56.4000
Epoch [4800/10000], Loss: 56.4000
Epoch [4900/10000], Loss: 56.4000
Epoch [5000/10000], Loss: 56.4000
Epoch [5100/10000], Loss: 56.4000
Epoch [5200/10000], Loss: 56.4000
Epoch [5300/10000], Loss: 56.4000
Epoch [5400/10000], Loss: 56.4000
Epoch [5500/10000], Loss: 56.4000
Epoch [5600/10000], Loss: 56.4000
Epoch [5700/10000], Loss: 56.4000
Epoch [5800/10000], Loss: 56.4000
Epoch [5900/10000], Loss: 56.4000
Epoch [6000/10000], Loss: 56.4000
Epoch [6100/10000], Loss: 56.4000
Epoch [6200/10000], Loss: 56.4000
Epoch [6300/10000], Loss: 56.4000
Epoch [6400/10000], Loss: 56.4000
Epoch [6500/10000], Loss: 56.4000
Epoch [6600/10000], Loss: 56.4000
Epoch [6700/10000], Loss: 56.4000
Epoch [6800/10000], Loss: 56.4000
Epoch [6900/10000], Loss: 56.4000
Epoch [7000/10000], Loss: 56.4000
Epoch [7100/10000], Loss: 56.4000
Epoch [7200/10000], Loss: 56.4000
Epoch [7300/10000], Loss: 56.4000
Epoch [7400/10000], Loss: 56.4000
Epoch [7500/10000], Loss: 56.4000
Epoch [7600/10000], Loss: 56.4000
Epoch [7700/10000], Loss: 56.4000
Epoch [7800/10000], Loss: 56.4000
Epoch [7900/10000], Loss: 56.4000
Epoch [8000/10000], Loss: 56.4000
Epoch [8100/10000], Loss: 56.4000
Epoch [8200/10000], Loss: 56.4000
Epoch [8300/10000], Loss: 56.4000
Epoch [8400/10000], Loss: 56.4000
Epoch [8500/10000], Loss: 56.4000
Epoch [8600/10000], Loss: 56.4000
Epoch [8700/10000], Loss: 56.4000
Epoch [8800/10000], Loss: 56.4000
Epoch [8900/10000], Loss: 56.4000
Epoch [9000/10000], Loss: 56.4000
Epoch [9100/10000], Loss: 56.4000
Epoch [9200/10000], Loss: 56.4000
Epoch [9300/10000], Loss: 56.4000
Epoch [9400/10000], Loss: 56.4000
Epoch [9500/10000], Loss: 56.4000
Epoch [9600/10000], Loss: 56.4000
Epoch [9700/10000], Loss: 56.4000
Epoch [9800/10000], Loss: 56.4000
Epoch [9900/10000], Loss: 56.4000
Epoch [10000/10000], Loss: 56.4000
Least-square error on training data (first 10 and last 10): 56.4000
Least-square error on test data (10 held out middle data points): 13.4000
```

There was much higher error on the training data than on the test data, which indicates that the training data was not a good sample of the overall data pattern. 

#### Question 2: MNIST Data

Question 2 is regarding neural network performance vs classifiers on the MNIST dataset. 

Below, we can see the accuracy of both the neural networks and the classifiers we are working with. 

```
Feed-forward neural network accuracy: 0.9733
LSTM Accuracy: 0.9862857142857143
SVM accuracy: 0.9764
Decision tree accuracy: 0.8696
```

Overall, we can see that LSTM has the greatest accuracy, FFNN and SVM have similar accuracies, and the decision tree has a markedly worse accuracy by around 10%. 

### Sec. V. Summary and Conclusion

In this homework, we explored the strengths and weaknesses of neural networks. 

It was found that FFNNs have a weak ability to classify data that is split like the second classification of data that splits up training data into the first 10 and last 10. Overall, the neural network performed worse than simple models like a line or a parabola for the 2D temperature data. 

When we fit the neural network to the MNIST data, we find that the model is extremely accurate. It gets around 97% accurate, which is comparable to the SVM accuracy. We found that LSTM (Long Short-Term Memory) has the highest accuracy rating however, with around 98.6% accuracy. The worst, by far, was the decision tree. 

Overall, we see that the FFNN performed better with more multi-dimensional data like the MNIST dataset. It most likely ends up overfitting, explaining why the training accuracy was so much better than the test accuracy for the first configuration of the temperature data. When we broke up the training data so much by splitting the training into the first 10 and the last 10 for the second configuration, we found that this kind of split causes the training accuracy to go down significantly, and the test accuracy to actually become better than the training accuracy. The FFNN worked much better when we gave it higher dimensional data like the MNIST dataset with 10 possible outputs with the 10 possible digits 0-9. 


---
# Homework 5 Writeup
## By: Gerin George



* [Abstract](https://github.com/gering92/EE399-Work/blob/main/README.md#abstract-4)
* [Sec. I. Introduction and Overview](https://github.com/gering92/EE399-Work/blob/main/README.md#sec-i-introduction-and-overview-4)
* [Sec. II. Theoretical Background](https://github.com/gering92/EE399-Work/blob/main/README.md#sec-ii-theoretical-background-4)
* [Sec. III. Algorithm Implementation and Development](https://github.com/gering92/EE399-Work/blob/main/README.md#sec-iii-algorithm-implementation-and-development-4)
* [Sec. IV. Computational Results](https://github.com/gering92/EE399-Work/blob/main/README.md#sec-iv-computational-results-4)
* [Sec. V. Summary and Conclusion](https://github.com/gering92/EE399-Work/blob/main/README.md#sec-v-summary-and-conclusion-3)



### Abstract

In this homework assignment, we are exploring neural networks capability to predict future states in the Lorenz equations, a system of differential equations that exhibit chaotic behavior under certain conditions. More specifically, we are training these models to advance the solution from time t to t + ∆t, using ρ values of 10, 28, and 40. The trained models were subsequently tested for their predictive performance at ρ values of 17 and 35.

The specific neural network architectures we are using in this homework assignment are Feed-Forward Neural Networks (FFNN), Long Short-Term Memory networks (LSTM), Recurrent Neural Networks (RNN), and Echo State Networks (ESN).

Our objective in this assignment is to determine which of these neural networks was best suited for predicting the complex, chaotic behavior of the Lorenz equations.

### Sec. I. Introduction and Overview

Neural networks are a powerful tool for modeling complex data structures and solving an expansive array of tasks. From image recognition to natural language processing, these models are able to make predictions. The underlying strength of neural networks is their ability to capture non-linear relationships in data, making them particularly suitable for modeling complex systems.

In this assignment, we ask neural networks to predict the future states of a system based on their training. The system in question is governed by the Lorenz equations, a set of differential equations which exhibit chaotic behavior under certain conditions. The Lorenz system is simple, but it is capable of generating complex and seemingly random patterns that have made it a canonical example of chaotic behavior in deterministic systems. 

The assignment is divided into two main parts. In the first part, we train a Feed Forward Neural Network to advance the solution of the Lorenz equations from time t to t + ∆t using ρ values of 10, 28, and 40. This training phase enables the network to learn the underlying dynamics of the system for these specific parameters. Subsequently, we test the network's ability to predict future states of the system at ρ values of 17 and 35, which were not included in the training phase. This allows us to evaluate the generalizability of the model.

In the second part of the assignment, we expand our exploration to include different types of neural network architectures, namely Feed-Forward Neural Networks (FFNNs), Long Short-Term Memory networks (LSTMs), Recurrent Neural Networks (RNNs), and Echo State Networks (ESNs). Each of these architectures brings unique strengths to handling sequential data, and the goal is to investigate their performance in forecasting the dynamics of the Lorenz system.

By the end of this assignment, we aim to gain insights into the applicability and effectiveness of different neural network architectures in predicting the behavior of complex dynamical systems. This could potentially provide valuable guidance for future research and applications involving time series forecasting in non-linear dynamical systems.

### Sec. II. Theoretical Background

There are several key concepts we must discuss to get the full scope of this assignment. The main one that we should discuss is the theory behind the Lorenz equations. The Lorenz equations are a set of three nonlinear differential equations that describe the deterministic behavior of a simple model for atmospheric convection. The set of equations are as follows:

$\frac{{dx}}{{dt}} = \sigma(y - x)$

$\frac{{dy}}{{dt}} = x(\rho - z) - y$

$\frac{{dz}}{{dt}} = xy - \beta z$

The variables x, y, and z make up the system state, t is the time, $\sigma$ (sigma), $\rho$ (rho), and $\beta$ (beta) are system parameters. For certain values of $\sigma$, $\rho$, and $\beta$, the Lorenz system exhibits chaotic behavior, meaning it is deterministic, but it shows a sensitive dependence on initial conditions. This is commonly referred to as the butterfly effect. This means that even a tiny change in the starting point can lead to vastly different outcomes, making long-term preidiction impossible. In this assignment, we are training neural networks to predict system states given three different $\rho$ values, then asking it to predict system states for two different $\rho$ values. We will naturally expect them to do quite poorly at this, but we are interested in seeing which NN performs best, and which NN performs worst. 

Now to get into the different neural network architectures that we will be working with in this homework assignment:

Feed Forward Neural Networks (FFNN):
Feed Forward Neural Networks are the foundation of deep learning. They consist of an input layer, one or more hidden layers, and an output layer. In FNNs, information flows in a unidirectional manner, from the input layer, through the hidden layers, to the output layer. Each layer is composed of interconnected nodes, called neurons, which apply a nonlinear activation function (ReLU) to the weighted sum of their inputs. FFNNs are typically used for tasks such as classification and regression. We should not expect FFNNs to be very good at predicting system states, because they do not have an explicit mechanism to handle sequential data. They can approximate the behavior to some extent, but they may not capture the chaotic dynamics and long-term dependencies of the Lorenz system as effectively as other models that are able to handle sequential data. 

Recurrent Neural Networks (RNN): 
RNNs are a class of neural networks specifically designed for sequential data processing. RNNs have feedback connections, which allow information to be passed from previous time steps to the current time step. This enables them to model temporal dependencies in data. RNNs maintain an internal hidden state that is updated at each time step and serves as a memory of past information. They can process inputs of variable lengths, and are widely used in tasks such as natural language processing, speech recognition, and sequence generation. RNNs are expected to fare better than FFNNs at predicting the states of the Lorenz equations due to being capable of capturing sequential dependencies. 

Long Short Term Memory (LSTM):
Long Short-Term Memory is a type of RNN architecture that is designed to overcome the vanishing gradient problem and capture long-term dependences in sequential data. LSTMs have a more complex structure when compared to RNNs, incorporating memory cells, input gates, forget gates, and output gates. LSTMs are therefore able to selectively retain or forget information over multiple time steps, making them effective in tasks involving sequential data such as this. LSTMs have the potential to capture the chaotic Lorenz equation system's dynamics and predict its states accurately. 

Echo State Networks (ESN):
Echo State Networks are also a type of recurrent neural network with a unique architecture that emphasizes the role of the reservoir, a fixed random network of recurrently connected neurons. ESNs are trained by adjusting only the readout layer weights, while keeping the reservoir weights fixed. The reservoir acts as a dynamic memory, transforming input signals into high-dimensional representations. ESNs are particularly efficient for processing time-varying inputs. The inherent reservoir structure of ESNs allow them to capture the complex behavior of the Lorenz equations systems. ESNs are particularly suitable for chaotic and nonlinear systems. We shoudl expect ESNs to be good at predicting the Lorenz system's dynamics well. 

### Sec. III. Algorithm Implementation and Development

#### Question 1: FFNN Trained to Predict Lorenz System States

To implement the FFNN, a custom class was created using the code below:

```python
# Define the model
class FFNNModel(nn.Module):
    def __init__(self):
        super(FFNNModel, self).__init__()
        self.fc1 = nn.Linear(3, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 3)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
```

To train and test the FFNN, a custom function was used called train_and_test_model. This function takes in the $\rho$ training values (10, 28, and 40), and the $\rho$ test values (17 and 35). Inside this function, we call a generate_lorenz_data function to actually create the data values to train the model on. We then in the same train_and_test_model function test the model. We use MSE to calculate the loss.

Below is the code for the train_and_test_model function: 

```python
def train_and_test_model(rho_train_values, rho_test_values):
    print("Generating training data...")
    # Generate training data
    nn_input = []
    nn_output = []
    for rho in rho_train_values:
        temp_input, temp_output = generate_lorenz_data(rho)
        nn_input.append(temp_input)
        nn_output.append(temp_output)
    nn_input = np.vstack(nn_input)
    nn_output = np.vstack(nn_output)
    print("Training data generated.")
    
    # Convert numpy arrays to PyTorch tensors
    nn_input = torch.from_numpy(nn_input).float()
    nn_output = torch.from_numpy(nn_output).float()

    # Create model instance
    model = FFNNModel()

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0008)

    print("Training model...")
    # Train the model
    num_epochs = 100
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = model(nn_input)
        loss = criterion(outputs, nn_output)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}, loss = {loss.item():.6f}")
    print("Model training complete.")
    
    print("Testing model...")
    # Test the model
    for rho in rho_test_values:
        test_input, test_output = generate_lorenz_data(rho)
        test_input = torch.from_numpy(test_input).float()
        test_output = torch.from_numpy(test_output).float()

        model.eval()
        with torch.no_grad():
            future_state_predictions = []
            current_state = test_input[0:1]
            for _ in range(len(test_input) - 1):
                next_state_prediction = model(current_state)
                future_state_predictions.append(next_state_prediction)
                current_state = next_state_prediction

            future_state_predictions = torch.vstack(future_state_predictions)
        mse_loss = criterion(future_state_predictions, test_output[:-1]).item()
        print(f"Test MSE loss for rho = {rho}: {mse_loss:.6f}")
    print("Model testing complete.")
```

A seperate function is also used to generate the data from the lorenz equation given a specific $\rho$ value:

```python
def generate_lorenz_data(rho):
    print(f"Generating Lorenz data for rho = {rho}")
    dt = 0.01
    T = 8
    t = np.arange(0, T + dt, dt)
    beta = 8 / 3
    sigma = 10

    def lorenz_deriv(x_y_z, t0, sigma=sigma, beta=beta, rho=rho):
        x, y, z = x_y_z
        return [sigma * (y - x), x * (rho - z) - y, x * y - beta * z]

    np.random.seed(123)
    x0 = -15 + 30 * np.random.random((100, 3))

    x_t = np.asarray([integrate.odeint(lorenz_deriv, x0_j, t) for x0_j in x0])

    nn_input = np.zeros((100 * (len(t) - 1), 3))
    nn_output = np.zeros_like(nn_input)

    for j in range(100):
        nn_input[j * (len(t) - 1):(j + 1) * (len(t) - 1), :] = x_t[j, :-1, :]
        nn_output[j * (len(t) - 1):(j + 1) * (len(t) - 1), :] = x_t[j, 1:, :]

    print(f"Lorenz data generated for rho = {rho}")
    return nn_input, nn_output
```

#### Question 2: Comparison of FFNN, LSTM, ESN, and RNN at Predicting Lorenz System States

Before diving into each different neural network model, this is a common function that is used to generate the lorenz data for each different $\rho$ value. 

```python
def generate_data(rho_values):
    sigma = 10
    beta = 8/3
    dt = 0.02
    T = 4
    t = np.arange(0, T+dt, dt)

    def lorenz_deriv(x_y_z, t0, sigma=sigma, beta=beta, rho_current=None):
        x, y, z = x_y_z
        return [sigma * (y - x), x * (rho_current - z) - y, x * y - beta * z]

    nn_input = []
    nn_output = []

    for rho in rho_values:
        x0 = -15 + 30 * np.random.random((50, 3))
        x_t = np.asarray([integrate.odeint(lorenz_deriv, x0_j, t, args=(sigma, beta, rho)) for x0_j in x0])

        for j in range(50):
            nn_input.extend(x_t[j,:-1,:])
            nn_output.extend(x_t[j,1:,:])

    return np.array(nn_input), np.array(nn_output)
```

For each NN model, we obtain the training values by using this function, then convert them to torch tensors with the following code:

```python
train_rho_values = [10, 28, 40]
x_train, y_train = generate_data(train_rho_values)
x_train_torch = torch.tensor(x_train, dtype=torch.float32)
y_train_torch = torch.tensor(y_train, dtype=torch.float32)
```

##### LSTM: 

The LSTM model is defined using the code below:

```python
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, 1, self.hidden_dim).requires_grad_()
        c0 = torch.zeros(self.num_layers, 1, self.hidden_dim).requires_grad_()
        x = x.unsqueeze(0)  # Add an extra dimension for batching
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out[:, -1, :]) 
        return out
```

The model is first initialized using the code below:

```python
model = LSTMModel(3, 50, 3, 1)
```

Our criterion is defined as MSE, and we use the Adam optimizer.

```python
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
```

We train the model using a simple for loop to loop through 100 Epochs:

```python
for epoch in range(100):
    if epoch % 10 == 0:
        print("Epoch: ", epoch)
    optimizer.zero_grad()
    outputs = model(x_train_torch)
    loss = criterion(outputs, y_train_torch)
    if epoch % 10 == 0:
        print("Current loss: ", loss.item())
    loss.backward()
    optimizer.step()
```

Another for loop is used to go through the two different test $\rho$ values and test the LSTM accuracy on the test values:

```python
test_rho_values = [17, 35]

for rho in test_rho_values:
    print(f"\nGenerating test data for rho = {rho}...")
    x_test, y_test = generate_data([rho])  # Generate test data for this rho
    x_test_torch = torch.tensor(x_test, dtype=torch.float32)
    y_test_torch = torch.tensor(y_test, dtype=torch.float32)
    print(f"Test data generation for rho = {rho} completed.\n")

    print(f"Running prediction on test data for rho = {rho}...")
    y_pred = model(x_test_torch)
    print(f"Prediction for rho = {rho} completed.\n")

    print(f"Calculating MSE loss on test data for rho = {rho}...")
    mse_loss = criterion(y_pred, y_test_torch)
    print(f"Test MSE Loss for rho = {rho}: ", mse_loss.item())
```

##### ESN: 

For the most part, the ESN follows the exact same process as the LSTM above. Here is how the ESN module is defined: 

```python
class ESN(nn.Module):
    def __init__(self, input_dim, reservoir_dim, output_dim):
        super(ESN, self).__init__()
        self.reservoir_dim = reservoir_dim
        self.input_weights = nn.Parameter(torch.randn(input_dim, reservoir_dim) / np.sqrt(input_dim), requires_grad=False)
        self.reservoir_weights = nn.Parameter(torch.randn(reservoir_dim, reservoir_dim), requires_grad=False)
        self.output_weights = nn.Parameter(torch.zeros(reservoir_dim, output_dim))

    def forward(self, x):
        reservoir_state = torch.tanh(x @ self.input_weights + self.reservoir_state @ self.reservoir_weights)
        self.reservoir_state = reservoir_state
        return reservoir_state @ self.output_weights

    def reset_state(self):
        self.reservoir_state = torch.zeros(1, self.reservoir_dim)
```

It is initialized slightly differently when compared to the LSTM:

```python
print("Initializing the model...")
model = ESN(3, 50, 3)
model.reset_state()
print("Model initialized.\n")
```

But the ESN still uses MSE for the loss, and it uses Adam as the optimizing method. Two for loops are also still used to train and test the model:

```python
for epoch in range(100):
    if epoch % 10 == 0:
        print("Epoch: ", epoch)
    optimizer.zero_grad()
    model.reset_state()
    outputs = model(x_train_torch)
    loss = criterion(outputs, y_train_torch)
    if epoch % 10 == 0:
        print("Current loss: ", loss.item())
    loss.backward()
    optimizer.step()
    
for rho in test_rho_values:
    print(f"\nGenerating test data for rho = {rho}...")
    x_test, y_test = generate_data([rho])  # Generate test data for this rho
    x_test_torch = torch.tensor(x_test, dtype=torch.float32)
    y_test_torch = torch.tensor(y_test, dtype=torch.float32)
    print(f"Test data generation for rho = {rho} completed.\n")

    print(f"Resetting model state and running prediction on test data for rho = {rho}...")
    model.reset_state()
    y_pred = model(x_test_torch)
    print(f"Prediction for rho = {rho} completed.\n")

    print(f"Calculating MSE loss on test data for rho = {rho}...")
    mse_loss = criterion(y_pred, y_test_torch)
    print(f"Test MSE Loss for rho = {rho}: ", mse_loss.item())
```

##### RNN:

Like the ESN and the LSTM, the RNN follows a very similar overall pattern. The RNN is defined using a module as defined below:

```python
class SimpleRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleRNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.rnn = nn.RNN(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        return out
```

The RNN model is initialized using the code below:

```python
model = SimpleRNN(3, 50, 3)
```

The same criterion is used for the RNN as the ESN, LSTM, and FFNN. The optimizer remains the Adam optimizing method, and a two for loops are used to train and test the model. 

### Sec. IV. Computational Results

The specific loss values for each neural network is hard to quantify, as it changes slightly every time it is run. So any hard values in this write up are not going to be exactly like what the values when the code is ran again. For this reason, we will not be comparing with direct values, but only looking at the general result and whether the loss for one NN was consistently higher or lower than the other.

#### FFNN Results:

Here are the training loss values for the FFNN:

```
Epoch 10, loss = 209.141693
Epoch 20, loss = 115.481621
Epoch 30, loss = 48.743965
Epoch 40, loss = 16.141109
Epoch 50, loss = 6.935210
Epoch 60, loss = 3.275884
Epoch 70, loss = 2.436378
Epoch 80, loss = 2.008651
Epoch 90, loss = 1.496558
Epoch 100, loss = 1.156174
```

```
Test MSE loss for rho = 17: 78.122292
```

```
Test MSE loss for rho = 35: 263.452789
```

#### LSTM Results:

The MSE loss for the LSTM training is printed below: 

```
Epoch:  0
Current loss:  307.0986328125
Epoch:  10
Current loss:  238.1931610107422
Epoch:  20
Current loss:  194.8776397705078
Epoch:  30
Current loss:  163.07229614257812
Epoch:  40
Current loss:  142.15342712402344
Epoch:  50
Current loss:  129.97702026367188
Epoch:  60
Current loss:  124.38225555419922
Epoch:  70
Current loss:  122.33444213867188
Epoch:  80
Current loss:  121.86066436767578
Epoch:  90
Current loss:  121.84431457519531
```

```
Test MSE Loss for rho = 17:  66.92090606689453
```

```
Test MSE Loss for rho = 35:  135.4553985595703
```

#### ESN Results:

The MSE loss for the ESN training is printed below: 

```
Epoch:  0
Current loss:  304.5351867675781
Epoch:  10
Current loss:  237.17877197265625
Epoch:  20
Current loss:  183.053466796875
Epoch:  30
Current loss:  142.16448974609375
Epoch:  40
Current loss:  113.07328796386719
Epoch:  50
Current loss:  93.4913101196289
Epoch:  60
Current loss:  80.96036529541016
Epoch:  70
Current loss:  73.31140899658203
Epoch:  80
Current loss:  68.84037017822266
Epoch:  90
Current loss:  66.31124877929688
```

```
Test MSE Loss for rho = 17:  28.195171356201172
```

```
Test MSE Loss for rho = 35:  67.7677001953125
```

#### RNN Results:

The MSE loss for the RNN training is printed below: 

```
Epoch:  0
Current loss:  317.39154052734375
Epoch:  10
Current loss:  231.94017028808594
Epoch:  20
Current loss:  186.25241088867188
Epoch:  30
Current loss:  153.58364868164062
Epoch:  40
Current loss:  134.9705352783203
Epoch:  50
Current loss:  126.64192199707031
Epoch:  60
Current loss:  123.91343688964844
Epoch:  70
Current loss:  123.38780212402344
Epoch:  80
Current loss:  123.3909912109375
Epoch:  90
Current loss:  123.40885925292969
```

```
Test MSE Loss for rho = 17:  65.30891418457031
```

```
Test MSE Loss for rho = 35:  130.9735107421875
```

#### Interpretation:

Overall, from the test MSE loss, we can see that there is a definite ranking that can be made for how each NN does when it comes to forecasting the system state dynamics. 

The ESN and LSTM are comparable, but the ESN has a clear lead when it comes to the accuracy. The LSTM is better than the RNN, and the FFNN is by far the worst.

### Sec. V. Summary and Conclusion

This homework assignment was valuable in order to more understand the strengths and weaknesses of the different neural networks we have discussed. Each of these neural networks uses a different method to learn, and as a result, have strengths and weaknesses that are different from each other. 

The Lorenz system is naturally very hard to predict as a result of its chaotic behavior, so for a NN to have a good accuracy, it needs to be good at capturing long-term dependencies, having memory and feedback connections, being able to handle nonlinear dynamics, and being robust to noise and initial conditions. Both ESNs and LSTMs, both a kind of RNN, have these features. It is clear from the output that the ESN and LSTM both perform the best. 

The ESN performs the best out of the 4 neural network architectures that we tested. ESNs are specifically designed to handle time-varying and chaotic system and their reservoir structure allows it to effectively capture the dynamics of the Lorenz system.

The LSTMs performed well due to being able to capture long-term dependencies in sequential data.

The RNNs are also capable of handling sequential data, but not as good as LSTMs for long-term dependencies. 

FFNNs are not explicitly designed for sequential data, which explains the high MSE loss for the test data. It struggles to accurately capture the cahotic dynamics and long term dependencies of the Lorenz system.

To recap, the overall ranking of the NNs for this task are below:

1. ESN
2. LSTM
3. RNN
4. FFNN


---
# Homework 6 Writeup
## By: Gerin George



* [Abstract](https://github.com/gering92/EE399-Work/blob/main/README.md#abstract-5)
* [Sec. I. Introduction and Overview](https://github.com/gering92/EE399-Work/blob/main/README.md#sec-i-introduction-and-overview-5)
* [Sec. II. Theoretical Background](https://github.com/gering92/EE399-Work/blob/main/README.md#sec-ii-theoretical-background-5)
* [Sec. III. Algorithm Implementation and Development](https://github.com/gering92/EE399-Work/blob/main/README.md#sec-iii-algorithm-implementation-and-development-5)
* [Sec. IV. Computational Results](https://github.com/gering92/EE399-Work/blob/main/README.md#sec-iv-computational-results-5)
* [Sec. V. Summary and Conclusion](https://github.com/gering92/EE399-Work/blob/main/README.md#sec-v-summary-and-conclusion-4)


### Abstract

This assignment presents an instructional guide on the application of SHallow REcurrent Decoder (SHRED) models to the analysis of a high-dimensional spatio-temporal field. Utilizing the weekly mean sea-surface temperature dataset by NOAA Optimum Interpolation SST V2, we explore how SHRED models can effectively reconstruct spatio-temporal fields using sensor measurements.

The SHRED model, a fusion of Long Short-Term Memory (LSTM) and Shallow Decoder Network (SDN), processes sensor measurement trajectories to yield valuable field data. The experiment proceeds by selecting three random sensor locations and assigning a trajectory length corresponding to one year of measurements.

Following data division into training, validation, and testing sets, the data is preprocessed using MinMaxScaler, generating corresponding input/output pairs. The SHRED model is then trained using the datasets, and reconstruction from the test set is subsequently evaluated for accuracy, determined by the mean square error relative to the ground truth.

The assignment encourages further exploration of the SHRED model's performance in relation to time lag variables, noise, and the number of sensors. The objective is to foster a deeper understanding of SHRED's effectiveness and limitations, contributing to its potential applications in spatio-temporal analysis.

### Sec. I. Introduction and Overview

The capacity to analyze high-dimensional spatio-temporal fields accurately is an invaluable tool in numerous scientific and technological domains. It has significant applications in meteorology, oceanography, geophysics, and many more fields, where it is crucial to track and predict the temporal changes of various physical quantities over large spatial areas. This assignment focuses on the use of SHallow REcurrent Decoder (SHRED) models, a novel combination of recurrent neural networks and shallow decoder networks, in the analysis of these fields.

The assignment begins with a brief introduction to SHRED models, explaining their underpinnings and purpose. SHRED models merge the capabilities of Long Short-Term Memory (LSTM) networks and Shallow Decoder Networks (SDN) to reconstruct high-dimensional spatio-temporal fields from sensor measurements. The principle is that the LSTM layer captures temporal correlations in the data while the SDN layer handles spatial correlations.

Three sensor locations are randomly selected from the dataset, with a trajectory length of 52 weeks, equivalent to a full year of measurements. The data is then partitioned into three subsets: training, validation, and testing, following which it is preprocessed with sklearn's MinMaxScaler for better model performance.

The SHRED model is trained on the training dataset and evaluated on the validation dataset. Finally, the model is used to generate reconstructions from the test set. The accuracy of these reconstructions is assessed by comparing them to the ground truth using the mean square error (MSE).

In addition to the primary focus of SHRED model application, the assignment also delves into examining the model's performance concerning the time lag variable, noise introduction, and the number of sensors. These explorations provide crucial insights into the performance and robustness of SHRED models in various scenarios.

This assignment serves as an instructive and exploratory endeavor into SHRED models and their potential in reconstructing high-dimensional spatio-temporal fields. It fosters an understanding of the models' applicability and adaptability, setting the groundwork for future applications and enhancements.


### Sec. II. Theoretical Background

The central focus of this assignment is the SHallow REcurrent Decoder (SHRED) model, a novel network architecture that combines the power of recurrent neural networks, specifically Long Short-Term Memory (LSTM), and a Shallow Decoder Network (SDN). The SHRED model is designed to reconstruct high-dimensional spatio-temporal fields from sensor measurements.

SHRED Model
The SHRED model architecture consists of two main components: an LSTM and an SDN. The LSTM network, with its inherent ability to learn and understand long-term dependencies in sequences, is used for capturing the temporal correlations within the dataset. This makes it highly suitable for processing a trajectory of sensor measurements taken over a period. On the other hand, the SDN is a feedforward neural network that handles the spatial correlations in the data, enabling it to handle high-dimensional fields effectively.

Formally, the SHRED architecture can be written as:

\begin{equation}
\mathcal {H} \left( \{ y_i \} _{i=t-k}^t \right) = \mathcal {F} \left( \mathcal {G} \left( \{ y_i \} _{i=t-k}^t \right) ; W_{RN} \right) ; W_{SD}
\end{equation}

where:
\begin{itemize}
\item $\mathcal {H}$ represents the SHRED model.
\item $\{ y_i \} _{i=t-k}^t$ denotes a trajectory of sensor measurements of a high-dimensional spatio-temporal field $\{ x_i \} _{i=t-k}^t$.
\item $\mathcal {F}$ is a feed-forward network (the SDN) parameterized by weights $W_{SD}$.
\item $\mathcal {G}$ is an LSTM network parameterized by weights $W_{RN}$.
\end{itemize}

Long Short-Term Memory (LSTM)
LSTM, a type of recurrent neural network, is designed to learn and understand long-term dependencies in sequential data. It employs gate mechanisms, including an input gate, forget gate, and output gate, to control the flow of information through its memory cell. This mechanism enables the LSTM to decide what information to store, delete, or output, making it capable of handling complex time-series data.

Shallow Decoder Network (SDN)
SDNs are a type of feedforward neural network that serve to handle the spatial dimension in the data. Unlike deep neural networks that consist of several layers of neurons, an SDN is a "shallow" network, meaning it has fewer hidden layers. This structure makes the SDN adept at approximating any continuous function, making it suitable for handling high-dimensional spatial data.

### Sec. III. Algorithm Implementation and Development

### Sec. IV. Computational Results

### Sec. V. Summary and Conclusion
