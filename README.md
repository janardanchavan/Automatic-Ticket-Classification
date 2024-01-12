# Automatic-Ticket-Classification using NLP

In this case study, we will create a model that can automatically classify customer complaints based on the products and services that the ticket mentions.

## Table of Contents
* [General Info and Problem Statement](#general-info-and-problem-statement)
* [Business goal](#business-goal)
* [Major Tasks](#major-tasks-to-be-performed)
* [Database](#database-link)
* [Technologies Used](#technologies-used)
* [Conclusions](#conclusions)
* [Acknowledgements](#acknowledgements)
* [Contacts](#contact)

## General Info and Problem statement

For a financial company, customer complaints carry a lot of importance, as they are often an indicator of the shortcomings in their products and services. If these complaints are resolved efficiently in time, they can bring down customer dissatisfaction to a minimum and retain them with stronger loyalty. This also gives them an idea of how to continuously improve their services to attract more customers. 

These customer complaints are unstructured text data; so, traditionally, companies need to allocate the task of evaluating and assigning each ticket to the relevant department to multiple support employees. This becomes tedious as the company grows and has a large customer base.

In this case study, as an NLP engineer for a financial company that wants to automate its customer support tickets system. As a financial company, the firm has many products and services such as credit cards, banking and mortgages/loans. 

## Business goal

We will be building a model that is able to classify customer complaints based on the products/services. By doing so, we can segregate these tickets into their relevant categories and, therefore, help in the quick resolution of the issue.

With the help of non-negative matrix factorization (NMF), an approach under topic modelling, we will be detecting patterns and recurring words present in each ticket. This can be then used to understand the important features for each cluster of categories. By segregating the clusters, we will be able to identify the topics of the customer complaints. 

* Credit card / Prepaid card

* Bank account services

* Theft/Dispute reporting

* Mortgages/loans

* Others 

With the help of topic modelling, we will be able to map each ticket onto its respective department/category. We can then use this data to train any supervised model such as logistic regression, decision tree or random forest. Using this trained model, we can classify any new customer complaint support ticket into its relevant department.

## Major tasks to be performed

1. Data loading

2. Text preprocessing

3. Exploratory data analysis (EDA)

4. Feature extraction

5. Topic modelling 

6. Model building using supervised learning

7. Model training and evaluation

8. Model inference

## Database link: 
[Click here](https://drive.google.com/file/d/1Y4Yzh1uTLIBLnJq1_QvoosFx9giiR1_K/view?usp=sharing) to download the database.

## Technologies Used
- Python 3.x version
- Python libraries 
    - json
    - numpy
    - pandas
    - re
    - nltk
    - spacy
    - string
    - en_core_web_sm
    - seaborn
    - matplotlib.pyplot
    - plotly.offline -> plot
    - plotly.graph_objects
    - plotly.express
    - sklearn.feature_extraction.text -> CountVectorizer, TfidfVectorizer
    - pprint -> pprint
    - google.colab -> drive
    - wordcloud -> WordCloud
    - sklearn.decomposition -> NMF
    - sklearn.feature_extraction.text -> TfidfTransformer
    - sklearn.model_selection -> train_test_split
    - sklearn.naive_bayes -> MultinomialNB
    - sklearn.linear_model -> LogisticRegression
    - sklearn.tree -> DecisionTreeClassifier
    - sklearn.ensemble -> RandomForestClassifier
    - sklearn.model_selection -> StratifiedKFold
    - sklearn.model_selection -> cross_val_score
    - sklearn.model_selection -> GridSearchCV
    - sklearn -> metrics
    - sklearn.metrics -> roc_auc_score
    - sklearn.metrics -> accuracy_score
    - sklearn.metrics -> precision_score
    - sklearn.metrics -> recall_score
    - sklearn.metrics -> precision_recall_fscore_support
    - sklearn.metrics -> classification_report
    - sklearn.metrics -> confusion_matrix
    - sklearn.metrics -> ConfusionMatrixDisplay

## Conclusions
- Based on the analysis we can see that the best model is Logistic Regression with GridSearchCG in our NLP scenario.

## Acknowledgements
- This project is a part of upGrad IIITB, Bangalore PG Program (ML & AI) Project work.

## Contact
Created by [Janardan Chavan](https://github.com/janardanchavan) and [Amina Bano](https://github.com/Amina-ban0) - feel free to contact us!
