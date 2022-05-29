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


### Naive Random Oversampling VS SMOTE Oversampling:

In this section, foll0wing metrics will be  provided in order to discover which algorithm results in the best performance between <strong>Naive random oversampling algorithm</strong> and <strong>the SMOTE algorithm</strong>

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
    <td> <ul>
        <li>Accuracy score: 0.65</li>
        <li>Precision
             <ul><li> High risk: 0.01</li>
               <li>Low risk: 1.00</li></ul></li>
        <li>Recall
             <ul><li> High risk: 0.61</li>
               <li>Low risk: 0.69</li></ul></li>
             </ul>
    </td>
   
  </tr>
  
  <tr>
    <td><img width="95%"src="https://user-images.githubusercontent.com/98676400/170883607-7eb4241b-6fdb-46cd-8630-98defc1c7418.png"</img>
    </td>
    <td><img width="95%"src="https://user-images.githubusercontent.com/98676400/170883629-4099a1a5-3fca-42ce-bbc8-ac54e656f13a.png"</img>
    </td>
  </tr>
  
  <tr>
    <td><img width="95%"src="https://user-images.githubusercontent.com/98676400/170883658-562cf2ee-6484-4064-ad9b-04d2ab3f3fca.png"</img>
    </td>
     <td><img width="95%"src="https://user-images.githubusercontent.com/98676400/170883792-edb525fb-7151-46f1-a86f-ef5ba51b9cbc.png"</img>
    </td>

   

  </tr>
</table>

### Cluster Centroids Undersampling VS SMOTEENN :
In this section,  following metrics will be  provided in order to discover which algorithm results in the best performance between <strong>Cluster Centroids</strong> undersampling and <strong> SMOTEENN</strong>.
1. Calculate the balanced accuracy score from <code>sklearn.metrics</code>
2. Calculate the confusion matrix from <code>sklearn.metrics</code>
3. Generate a classication report using the  <code>imbalanced_classification_report </code> from <code>imbalanced-learn</code>.

<table>
  <tr>
    <th>Cluster Centroids Undersampling </th>
    <th>SMOTEENN Combination (Over and Under) Sampling </th>
  </tr>
  <tr>
    <td> <ul>
        <li>Accuracy score: 0.54</li>
        <li>Precision
             <ul><li> High risk: 0.01</li>
               <li>Low risk: 1.00</li></ul></li>
        <li>Recall
             <ul><li> High risk: 0.69</li>
               <li>Low risk: 0.40</li></ul></li>
             </ul>
    </td>
    <td> <ul>
        <li>Accuracy score: 0.64</li>
        <li>Precision
             <ul><li> High risk: 0.01</li>
               <li>Low risk: 1.00</li></ul></li>
        <li>Recall
             <ul><li> High risk: 0.71</li>
               <li>Low risk: 0.57</li></ul></li>
             </ul>
    </td>
   
  </tr>
  <tr>
    <td><img width="95%"src="https://user-images.githubusercontent.com/98676400/170883247-8f46ae17-ab31-4b26-b558-88fcf6471e3f.png"</img>
</td>
    <td><img width="95%"src="https://user-images.githubusercontent.com/98676400/170883266-0f1a493e-3cc5-4a12-8750-192ecc7c6244.png"</img></td>

  </tr>
   <tr>
    <td><img width="95%"src="https://user-images.githubusercontent.com/98676400/170883388-96002873-c986-431c-8691-1668731c36af.png"</img>
    </td>
    <td><img width="95%"src="https://user-images.githubusercontent.com/98676400/170883406-9bf189e5-c707-4f4e-9579-d0f94192ea31.png"</img></td>
 
  </tr>
</table>

### Balanced Random Forest Classifier VS Easy Ensemble AdaBoost Classifier:

In this section, foll0wing metrics will be  provided in order to discover which algorithm results in the best performance between <strong>Balanced Random Forest Classifier</strong> and <strong>Easy Ensemble AdaBoost Classifier</strong>

1. Calculate the balanced accuracy score from <code>sklearn.metrics</code>
2. Calculate the confusion matrix from <code>sklearn.metrics</code>
3. Generate a classication report using the  <code>imbalanced_classification_report </code> from <code>imbalanced-learn</code>.

<table>
  <tr>
    <th>Balanced Random Forest Classifier </th>
    <th>Easy Ensemble AdaBoost Classifier </th>
  </tr>
  <tr>
    <td> <ul>
        <li>Accuracy score: 0.79</li>
        <li>Precision
             <ul><li> High risk: 0.03</li>
               <li>Low risk: 1.00</li></ul></li>
        <li>Recall
             <ul><li> High risk: 0.70</li>
               <li>Low risk: 0.87</li></ul></li>
             </ul>
    </td>
    <td> <ul>
        <li>Accuracy score: 0.93</li>
        <li>Precision
             <ul><li> High risk: 0.09</li>
               <li>Low risk: 1.00</li></ul></li>
        <li>Recall
             <ul><li> High risk: 0.92</li>
               <li>Low risk: 0.94</li></ul></li>
             </ul>
    </td>
   
  </tr>
  <tr>
    <td><img width="95%"src="https://user-images.githubusercontent.com/98676400/170884277-ac585c51-d01d-48fe-bd29-0d11c66d2cf8.png"</img>
    </td>
    <td><img width="95%"src="https://user-images.githubusercontent.com/98676400/170884296-68f6ccf9-6e28-44d9-badb-72dc1ffa0676.png"</img>
    </td>
  </tr>
  <tr>
    <td><img width="95%"src="https://user-images.githubusercontent.com/98676400/170884323-12350b60-c3a2-4cdc-b728-b3de5f593720.png"</img>
    </td>
    <td><img width="95%"src="https://user-images.githubusercontent.com/98676400/170884336-52aaa365-51b1-438e-8d7e-625b0d5a9f6a.png"</img></td>

  </tr>
</table>








## Resources 

* Data Source: [ LoanStats_2019Q1.csv ](https://github.com/aktugchelekche/Credit_Risk_Analysis/blob/main/Resources/LoanStats_2019Q1.csv)
* Software/Languages: Jupyter Notebook- Google Colab, Python
* Libraries: Scikit-learn








