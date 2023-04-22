# *Assignment Writeup Links*

=================
* [Homework 1 Writeup](https://github.com/gering92/EE399-Work/blob/main/README.md#homework-1-writeup)
* [Homework 2 Writeup](https://github.com/gering92/EE399-Work/blob/main/README.md#homework-2-writeup)
* [Homework 3 Writeup](https://github.com/gering92/EE399-Work/blob/main/README.md#homework-3-writeup)

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

