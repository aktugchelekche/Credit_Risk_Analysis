# Credit Risk Analysis

Credit Risk Analysis with Machine Learning 

## Overview of the analysis

The purpose of this analyisis is to build several Machine Learning models and algoriths to predict credit risk for loan applications. After completion of this analyisis, approving or denying application for loan will be more efficient, accurate and also lower default rates. I will utilize Python and Scikit-learn libraries and several machine learning models to compare the ML models and determine how well each model classifies and predicts data.

# Results
In this project, I am utilizing following models and algorithms to find best prediction model for credit risk analysis:
* Oversampling Models    :  RandomOverSampler and SMOTE algorithms.
* Undersampling Model    :  ClusterCentroids algorithm.
* Combining Models       : SMOTEENN algorithm that combines the SMOTE and Edited Nearest Neighbors (ENN) algorithms. 
* Compareing Models      : BalancedRandomForestClassifier and EasyEnsembleClassifier.

After applying exploratory data analysis with <strong>Pandas and Numpy</strong> for the dataset, I am using the <strong>imbalanced-learn</strong> and </trong>scikit-learn </strong>libraries for evaluating three machine learning models by using resampling to determine which is better at predicting credit risk. 
I will start with the oversampling <strong> RandomOverSampler</strong> and <strong>SMOTE </strong>algorithms, and then use the undersampling <strong>ClusterCentroids</strong> algorithm. Using these algorithms, resample the dataset, view the count of the target classes, train a logistic regression classifier for :
* Calculate the balanced accuracy score.
* Generate a confusion matrix.
* Generate a classification report.

Note: A random state of 1 for each sampling algorithm to ensure consistency between tests. 


### Naive Random Oversampling VS SMOTE Oversampling

In this part, follwing metrics will be  provided in order to discover which algorithm results in the best performance between <strong>Naive random oversampling algorithm</strong> and <strong>the SMOTE algorithm</strong>

1. Calculate the balanced accuracy score from <code>sklearn.metrics</code>
2. Calculate the confusion matrix from <code>sklearn.metrics</code>
3. Generate a classication report using the  <code>imbalanced_classification_report </code> from <code>imbalanced-learn</code>.

<table>
  <tr>
    <th>Naive Random Oversampling </th>
    <th>SMOTE Oversampling</th>
  </tr>
  <tr>
    <td> <ul>
        <li>Accuracy score: 0.64</li>
        <li>Precision
             <ul><li> High risk: 0.01</li>
               <li>Low risk: 1.00</li></ul></li>
        <li>Recall
             <ul><li> High risk: 0.66</li>
               <li>Low risk: 0.62</li></ul></li>
             </ul>
    </td>
    <td>Maria Anders</td>
   
  </tr>
  <tr>
    <td><img src="https://user-images.githubusercontent.com/98676400/170881598-a3bae573-f7fe-4a89-bbd9-09af37992b6a.png"
</td>
    <td>Francisco Chang</td>
 
  </tr>
</table>















## Resources 

* Data Source: [ LoanStats_2019Q1.csv ](https://github.com/aktugchelekche/Credit_Risk_Analysis/blob/main/Resources/LoanStats_2019Q1.csv)
* Software/Languages: Jupyter Notebook- Google Colab, Python
* Libraries: Scikit-learn








