{
 "cells": [
  {
   "cell_type": "raw",
   "metadata": {},
   "source": [
    "// TOPICS \n",
    "Logistic Regression -\n",
    "1: How to use Linear Regression for classification? Why is it bad?\n",
    "2: Logistic regression & Sigmoid function\n",
    "3: Cost function\n",
    "4: Gradient Descent for Logistic Regression\n",
    "5: Multiclass classification\n",
    "6: Extended form by adding extra features & Regularization"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Using Linear Regression for classification"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Linear Regression , although is a regression algorithm but can be used in classification problems also. "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Lets consider an example where the results of linear regression are between 0 and 5 (real numbers like 3.567) and we need to classify the data into 6 categories i.e. 0,1,2,3,4,5 .A simple way of using this for classification is to just round off the result to the nearest integer between 0 and 5. So , if the result from linaer regression is [1.23 , 0.43 , 4.32 , 3.49] we get the results of our classification as [1,0,4,3] ."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "We need to set the thresholds for classification . In the above exapmle thresholds were : [0-0.5) for 0 , [0.5,1.5) for 1 , [1.5,2.5) for 2 and so on. You can decide your own threshold values depending on the data you have. "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Why is linear regression bad for classification?"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Linear Regression can be used for classification by defining appropriate threshold values , but it is not the right algorithm for classification problems because -\n",
    "1. Outliers can affect the best fit line and thus the decision boundary.\n",
    "2. Values predicted by Linear Regression will be continuous, whereas expected results will be discrete."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "With linear regression you fit a polynomial through the data - say, like in the example below we're fitting a straight line through {tumor size, tumor type} sample set:\n",
    "<img src=\"one.png\">\n",
    "Above, malignant tumors get 1 and non-malignant ones get 0, and the green line is our hypothesis h(x). To make predictions we may say that for any given tumor size x, if h(x) gets greater than 0.5 we predict malignant tumor, otherwise we predict benign."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "It seems that everytime we have the right answer according to the data above , right? Now lets change the data a little bit and for a quite large value of tumor size lets add a Malilgnant cancer data point. Now our line h(x) begins to look somewhat like this->\n",
    "<img src=\"three.png\">\n",
    "We can clearly see that now the predictions are not correct. Because we are trying to fit a line through the data we are getting , the line will be dependant on the quality and type of data we get. We cannot change the hypothesis each time a new sample arrives. Instead, we should learn it off the training set data, and then (using the hypothesis we've learned) make correct predictions for the data we haven't seen before. This can be done by creating a decision boundary."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Decision boundary is the boundary which seperates the two regions in classification. If we have a binary classification with values 0 and 1 then one side of this boundary will be 0 and the other will be 1. Take a look at the image below ->\n",
    "<img src=\"two.png\">\n",
    "Here, the purple colour line is the decision boundary. All points on the left side of this line correspond to 0 and points on the right side correspond to 1. In the above problem (and also many other classification problems) we more importantly want that the points are placed on the correct side of the decision boundary and not on how far they are on the correct side . What we mean is that we are somewhat okay if the points are quite close to the decision boundary , as long as they are on the right side of it."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "The above decision boundary (purple line) is similar to the one which would have been generated if we would have used Logistic Regression in the above problem(Logistic Regression is dicussed in detail next). Both linear regression and logistic regression give you a straight line (or a higher order polynomial) but those lines have different meaning:<br>\n",
    "-- line for linear regression interpolates, or extrapolates, the output and predicts the value for x we haven't seen.<br>\n",
    "-- h(x) for logistic regression tells you the measure (like probability) that x belongs to the \"positive\" class. Or you can see the line as the decision boundary."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Logistic Regression "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Logistic Regression is a statistical method for analyzing a dataset in which there are one or more independent variables that determine an outcome. Logistic Regression is actually a classification algorithm. Logistic regression is used to describe data and to explain the relationship between one dependent binary variable and one or more nominal, ordinal, interval or ratio-level independent variables.<br>\n",
    "Before starting with Logistic regression we need to know about a function called Sigmoid function and its properties."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Sigmoid Function"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "A sigmoid function is a mathematical function having an \"S\" shaped curve (sigmoid curve). Mathematically , the function is :\n",
    "<img src=\"S1.png\">"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "and its curve looks like :\n",
    "<img src=\"S2.png\">\n",
    "With its output ranging between 0 and 1. As we can clearly see that the the curve quickly goes toward 1 when t>0 and toward 0 when t<0 and at t=0 it is equal to 0.5. Value of the above function for t=2 is 0.88 and for t=-2 is 0.119 , which shows how sharply it goes towards 0 and 1."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Because of the property of sigmoid function to give output between 0 and 1 we can use its output like probability, but not exactly as probability. For example, the property of probability that P(true)+P(false)=1 may not be true is case of sigmoid function i.e. S(true)+S(false) may not be equal to 1."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "collapsed": true
   },
   "source": [
    "The logistic function has this further, important property, that its derivative can be expressed by the function itself (),\n",
    "<img src=\"S3.png\">"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "As at t=0 we have S(t)=0.5 , and for t>0 we have S(t)>0.5 (sharply rising to 1 so we consider it 1) and for t<0 we have S(t)<0.5 (sharply falling to 0 so we consider it 0) , we have our decision boundary as 0.5.\n",
    "For example, as our threshold is .5 and our prediction function returned .7, we would classify this observation as positive(1). If our prediction was .2 we would classify the observation as negative(0). For logistic regression with multiple classes we could select the class with the highest predicted probability."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<img src=\"S4.png\" width=\"450px\">"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## cost function of logistic regression"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Earlier in linear regression we have used the following cost function:"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "\\begin{equation*}\n",
    " \\left( \\sum_{k=1}^n (y_t - y_p)^2  \\right)\n",
    "\\end{equation*}\n",
    "\n",
    "where in y(predicted) was replaced by its value mx."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "\\begin{equation*}\n",
    " \\left( \\sum_{k=1}^n (y - mx)^2  \\right)\n",
    "\\end{equation*}"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "This cost function was of degree 2 and had only one minimum. But in the case of logistic regression if we use the same cost function then the cost function will end up having many minimas because of the hypothesis function of logistic regression. Therefore while estimating a minimum value we may end up at some local minimum rather than a global minimum.  \n",
    "\n",
    "\\begin{equation*}\n",
    " \\left( \\sum_{k=1}^n (y - {\\frac{1}{1+e^{-mx} }})^2  \\right)\n",
    "\\end{equation*}"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "So here we use CROSS ENTROPY to measure the cost of our model.\n",
    "<pre style=\"color:brown\">\n",
    "                cost = -log(h(x))   if y = 1\n",
    "                       -log(1-h(x)) if y = 0\n",
    "</pre>\n",
    "here h(x) is the hypothesis function for logistic regression(i.e sigmoid function) and y is the actual true output ,i.e the actual label , for the current considered values of features.\n",
    "\n",
    "This error function penalizes for the wrong predictions we make.\n",
    "\n",
    "For example, consider that y(the actual label) is 1 and we classify it as e(e is tending to zero), then our cost will be -log(e) i.e a very very high positive value, and the cost reduces as we classify it near to 1 . The cost becomes zero when we classify it correctly as 1 .\n",
    "Also consider that y(the actual label) is 0 and we classify it as t(nearly equal to 1) , then the cost for that will be -log(1-t) = -log(e), where e is tending to zero,and again the cost has a very very high positive value.And the cost tends to zero when we classify it correctly as 0.\n",
    "   "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<img src=\"log.png\" width=\"250\"/>"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "we can combine the two cases of our cost function as :\n",
    "<pre style=\"color:brown\">\n",
    "            cost = [-ylog(h(x))]-[(1-y)log(1-h(x))]\n",
    "</pre>\n",
    "\n",
    "The only parameter which we can vary in our cost function is h(x) and  therein the variable is m, as shown below. "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<img src=\"mwalah.jpeg\" width=\"400\"/>"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Hence our aim is to improve the accuracy of our model and reduce the cost by selecting a proper value(s) for m , and we do so by GRADIENT DESCENT as it was done in linear regression.\n",
    "\n",
    "Our cost function is a convex function i.e it has only one local minimum therefore we can use gradient descent approach to find the apt value of m ."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Multiclass classification"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Multiclass Classification , as the name suggests ,are the kind of problems in which using the given parameters/features we need to classify into <b>more than 2 classes</b>.\n",
    "<img src=\"M1.png\" width=\"500px\">\n",
    "<br>Instead of y=0,1 we will expand our definition so that y=0,1...n. Basically we re-run binary classification multiple times, once for each class. <b>For each sub-problem, we select one class (YES) and lump all the others into a second class (NO). Then we take the class with the highest predicted value.</b>\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Lets understand it better with the exapmle of Iris dataset where we have to categorise flowers in 3 categories based on four features.For more information regarding the iris dataset you can follow check the [link](https://archive.ics.uci.edu/ml/datasets/iris)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Lets consider the 3 categories as A,B and C.<br>\n",
    "We run binary classification(train) on this dataset with 3 different structures and meaning of the data -><br>\n",
    "1)  one time with A as true and others (B and C) as false.<br>\n",
    "2)  second with B as true and others (A and C) as false.<br>\n",
    "3)  third with C as true and others (A and B) as false.<br>\n",
    "<br>\n",
    "Now , when we get a test sample we pass it into the model. Let the output from the three structures as shown above be O1,O2 and O3. We classify the test sample in the class which has the highest value among these three outputs. For example, if O1=0.5 and O2=0.6 and O3=0.9 we classify the test sample as C.<br>\n",
    "Here we kind of use the output of Logistic regression as being the probability of the test sample being in that particular class and not in other classes. But as mentioned earlier also this measure is not strictly probability."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Creating useful features from given features and regularisation"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "A common practise is to use the extended form by creating extra features from the given set of features. One way of doing this is shown here . This is the most general way and is used widely .<br>\n",
    "Let's take an example in which we have 3 features f1 , f2 and f3. For these features we will have equation for input to sigmoid with at most one degree in f1,f2 and f3. By degree 1 we mean equations of the form <i>a(f1)+b(f2)+c(f3)+d</i> , where a,b,c and d are some coefficients. <br><br>\n",
    "What we plan to do is add more features such as f1\\*f1 , f2\\*f2 , f3\\*f3 , f1\\*f3 , f1\\*f2 and f2\\*f3 so that our equation can now be of 2nd degree. Main point to note here is that we have increased our number of features from just 3 to 9 ,but the newly added 6 features are derived from the already existing 3 features.<br><br>\n",
    "You can create and add features of any degree you fell like. f1\\*f2\\*f3  will make an equation of 3rd degree. The more features you add the better your decision boundary tries to fit in the training data. Generally we get better results with higher degree terms in the equation.<br>\n",
    "We must acknowledge the fact that if the dependancy of the output is very low on the a certain factor say f1\\*f2 then the model will assign a value to its coefficient which ensures that it has less effect in the output. <br>\n",
    "With higher degree features being added to the dataset we are now able to achieve boundaries of many shapes such as parabolic and even some complex shapes.<br><br>\n",
    "This addition of features comes with a cost. If we keep on adding more and more features of higher degree , our model to try to fit itself more to the training data and may cause the problem of **overfitting**."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**Overfitting** is a modeling error which occurs when a function is too closely fit to a limited set of data points. Due to this the model poorly respond to the unseen data or the training data. This is one of many reasons for a trained model to perform poorly on test data.<br>\n",
    "So what we actually need to have a trade off between the complexity of features we want to use and the extent to which we want our model to fit to the training data."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Consider the example below for more clarity:"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<img src=\"R1.png\" width=\"600px\">"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "In the above example we see that in the first figure when we use equation of degree one we just get a line which does not depicts the data nicely.<br>\n",
    "For the second figure we use equation of degree 2 as we have terms like x1\\*x2. The curve we obtain here is depicting the data quite well. This should be the optimal solution even though it has a pair of wrong classifications.\n",
    "In the third example we use equation with degree 5. We can clearly see that the decision boundary is trying to fit itself to the data. This is the case of over fitting."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "One way of reducing overfitting is by **regularisation**. A very simple explaination of regularisation can be that it discourages complex features in the dataset. Lets try to understand regularisation with an example.<br>Assume that we have (m1\\*x1) + (m2\\*x2) + (m3\\*x1\\*x1) + (m4\\*x2\\*x2) + (m5\\*x1\\*x2) as our input to the sigmoid function in case of Logistic regression. \n",
    "<br>\n",
    "We add this term to the cost function:"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "\\begin{equation*}\n",
    "\\beta \\left( \\sum_{k=1}^n (m_k)^2  \\right)\n",
    "\\end{equation*}"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "where β is the regularisation parameter and m are the coefficients given to n features.We actually are telling our model that the more importance you give to a particular feature the more it will add to the cost function. So ,this automatically acts as a deciding factor for the model. A particular feature will be give higher valued coefficient only if it is significantly important in deciding the outcome. This will naturally reduce overfitting."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "This type of regularisation in which we add square of the coefficient's value multiplied by β to the cost function is called **L2 Regularisatin** and if we just added the coefficients value multiplied by β to the cost function ,it is called **L1 Regularisation**. "
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.6.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
