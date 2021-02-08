# Project Charter

## Project Description

Social media sites like Twitter have become important parts of our public safety apparatus. In 2009 news that U.S Airways 1549 had landed in the Hudson was first broken by users of Twitter. Public safety institutions would benefit greatly from this real-time, crowd-sourced feed of tweets that may or may not indicate a disaster in progress. 

Our project will classify tweets as “relating to” or “not relating to” disasters with NLP. Specifically, we will take advantage of attention metrics to make our model context-aware. Afterwards, we will present a case-study of how this model would have performed during a past disaster. For example, we will scrape and classify tweets from the southeastern United States during the day of the Nashville tornados. We will then see if we can identify clusters of tweets tagged as “relating to” a disaster, thus proving our model could’ve automatically identified these tornados-in-progress.


## Scope
The main problem we are trying to solve is the classification of disaster Tweets based solely on the content of the Tweet.  Our solution will take in the Tweets and output a 1 if the Tweet is about a real disaster or a 0 if it is not.  Initially, this will be achieved with more naïve approaches in order to establish an accuracy baseline.  This includes word tokenization paired with traditional machine learning algorithms (support vector machine, random forest, boosting, etc.) and bag-of-words approaches.  However, these naïve approaches fail to leverage how words relate to each other in order to make their prediction.  The word content of the Tweets alone is not enough to classify Tweets with a high degree of accuracy because different words mean different things in different contexts. 
	
In order to account for this, we will use more advanced NLP algorithms on these Tweets to improve upon our baseline accuracy.  These advanced techniques include Word2vec, which uses neural networks to create word embeddings that preserve deeper meanings between words, and the Attention mechanism, which calculates the correlation between different words in a sentence in order to determine which words are related to each other in their usage.  These techniques are present in most state-of-the-art NLP models and will help in achieving peak performance on our classifier. 

Eventually, we want to take our model and apply it to a real-life disaster scenario.  Taking Hurricane Sandy as an example, we will first mine Twitter data from the time period right before and throughout the course of Hurricane Sandy.  Using the model built by analyzing the Twitter data from Kaggle, we will flag Tweets from our mined Twitter data that the algorithm determines to be about real disasters.  These flagged Tweets will allow us to create a heat map that tracks the path of the hurricane.  Our final visual will be a map that shows the flagged Tweets and their locations moving through time.


## Personnel
The team for the project consists of three members: Brendan Manning, Brendan Magdamo, and Stephanie Scheerer. The majority of the work will be completed as a team with no specific phases assigned to each team member in order to encourage collaboration and foster the best outcomes. Loose guidelines for areas that different team members will manage are as follows but are subject to change. 

Brendan Manning will manage the setup of an AWS cloud environment for collection of mass Twitter data for the phase three simulation and will determine the means of what cluster metric is best suited for the situation. Brendan Magdamo will manage exploration of various preprocessing that will be performed on the data as well as working on assorted NLP models for each phase of the project. Stephanie Scheerer will also work on iterative NLP models for phases one and two as well as managing the necessary documentation for the project. 

	
## Metrics
To quantify the success of the proposed Twitter Disaster Recognition project, the F1 score will be used to evaluate the produced models. The F1 score is a standard method for evaluating machine learning models and is the metric by which all submissions to the Kaggle competition are being evaluated. F1 is calculated using precision and recall which take into account the number of true/false positives and also the number of false negatives so it gives a holistic view of model performance that allows for avoidance of overfitting. The formula for the F1 score is shown in Figure 1. Iterating through multiple models is part of the project’s methodology, so increases in F1 score will be tracked through those iterations to illustrate improvements in the model and to track project progress.
![alt text](https://static.packt-cdn.com/products/9781785282287/graphics/B04223_10_02.jpg)


In addition to using the F1 score, some cluster evaluation will be performed to evaluate how well the final phase of the project performs at identifying the area of interest for a historical natural disaster. Common metrics for cluster evaluation include Purity, Normalized Mutual Information (NMI), and Rand Index. Purity is a method that focuses on examining the distinctiveness of each cluster and thus it cannot be used in assessing trade off between number of clusters and cluster quality. NMI utilizes an entropy calculation to determine cluster quality and is advantageous as it isn’t affected by differing numbers of clusters. Lastly Rand index is a method of determining similarity between two clusters using actual and predicted labels. Each of these metrics has advantages and disadvantages. It is also possible that a clustering metric specific to this project will be developed but this has not yet been determined. The decision of which metric to use will depend on the outcome of the final phase of the project and will be decided when that phase of the project is reached.



## Plan
* Phases (milestones), timeline, short description of what we'll do in each phase.

## Architecture
* Data
  * What data do we expect? Raw data in the customer data sources (e.g. on-prem files, SQL, on-prem Hadoop etc.)
* Data movement from on-prem to Azure using ADF or other data movement tools (Azcopy, EventHub etc.) to move either
  * all the data, 
  * after some pre-aggregation on-prem,
  * Sampled data enough for modeling 

* What tools and data storage/analytics resources will be used in the solution e.g.,
  * ASA for stream aggregation
  * HDI/Hive/R/Python for feature construction, aggregation and sampling
  * AzureML for modeling and web service operationalization
* How will the score or operationalized web service(s) (RRS and/or BES) be consumed in the business workflow of the customer? If applicable, write down pseudo code for the APIs of the web service calls.
  * How will the customer use the model results to make decisions
  * Data movement pipeline in production
  * Make a 1 slide diagram showing the end to end data flow and decision architecture
    * If there is a substantial change in the customer's business workflow, make a before/after diagram showing the data flow.

## Communication
* How will we keep in touch? Weekly meetings?
* Who are the contact persons on both sides?
