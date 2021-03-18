# Project Charter

## Project Description

Social media sites like Twitter have become important parts of our public safety apparatus. In 2009 news that U.S Airways 1549 had landed in the Hudson was first broken by users of Twitter. Public safety institutions would benefit greatly from this real-time, crowd-sourced feed of tweets that may or may not indicate a disaster in progress. 

Our project will classify tweets as “relating to” or “not relating to” disasters with NLP. Specifically, we will take advantage of attention metrics to make our model context-aware. Afterwards, we will present a case-study of how this model would have performed during a past disaster. For example, we will scrape and classify tweets from the southeastern United States during the day of the Nashville tornados. We will then see if we can identify clusters of tweets tagged as “relating to” a disaster, thus proving our model could’ve automatically identified these tornados-in-progress.


## Scope
The main problem we are trying to solve is the classification of disaster Tweets based solely on the content of the Tweet.  Our solution will take in the Tweets and output a 1 if the Tweet is about a real disaster or a 0 if it is not.  Initially, this will be achieved with more naïve approaches in order to establish an accuracy baseline.  This includes word tokenization paired with traditional machine learning algorithms (support vector machine, random forest, boosting, etc.) and bag-of-words approaches.  However, these naïve approaches fail to leverage how words relate to each other in order to make their prediction.  The word content of the Tweets alone is not enough to classify Tweets with a high degree of accuracy because different words mean different things in different contexts. 
	
In order to account for this, we will use more advanced NLP algorithms on these Tweets to improve upon our baseline accuracy.  These advanced techniques include the Attention mechanism, which calculates the correlation between different words in a sentence in order to determine which words are related to each other in their usage, and a BERT language model, which is a version of state-of-the-art language models.  These techniques are present in most state-of-the-art language pipelines and allow us to make more accurate predictions pertaining to sentiment analysis.  BERT models can by implemented manually or can be used in Simple Transformers, where only the text and the labels are taken in as inputs for training. 

Eventually, we want to take our model and apply it to a real-life disaster scenario.  Taking Hurricane Sandy as an example, we will first mine Twitter data from the time period right before and throughout the course of Hurricane Sandy.  Using the model built by analyzing the Twitter data from Kaggle, we will flag Tweets from our mined Twitter data that the algorithm determines to be about real disasters.  These flagged Tweets will allow us to create a heat map that tracks the path of the hurricane.  Our final visual will be a map that shows the flagged Tweets and their locations moving through time.


## Personnel
The team for the project consists of three members: Brendan Manning, Brendan Magdamo, and Stephanie Scheerer. The majority of the work will be completed as a team with no specific phases assigned to each team member in order to encourage collaboration and foster the best outcomes. Loose guidelines for areas that different team members will manage are as follows but are subject to change. 

Brendan Manning will manage the setup a Twitter scraping for the phase three simulation and will determine the means of what cluster metric is best suited for the situation. He will also focus on optimizing a language model and testing different classification algorithms.  Brendan Magdamo will manage exploration of various preprocessing that will be performed on the data as well as working on assorted NLP models for each phase of the project. Stephanie Scheerer will use the Twitter scraper to mine Tweets from different cities surrounding several disasters and help with maintaining documentation. 

	
## Metrics
To quantify the success of the proposed Twitter Disaster Recognition project, the F1 score will be used to evaluate the produced models. The F1 score is a standard method for evaluating machine learning models and is the metric by which all submissions to the Kaggle competition are being evaluated. F1 is calculated using precision and recall which take into account the number of true/false positives and also the number of false negatives so it gives a holistic view of model performance that allows for avoidance of overfitting. The formula for the F1 score is shown in Figure 1. Iterating through multiple models is part of the project’s methodology, so increases in F1 score will be tracked through those iterations to illustrate improvements in the model and to track project progress.

![alt text](https://github.com/bmagdamo1/DisasterTweets/blob/main/Docs/Project/F1.png?raw=true)   
Figure 1:  Steps for calculation of F1.

In addition to using the F1 score, some cluster evaluation will be performed to evaluate how well the final phase of the project performs at identifying the area of interest for a historical natural disaster. Common metrics for cluster evaluation include Purity, Normalized Mutual Information (NMI), and Rand Index. Purity is a method that focuses on examining the distinctiveness of each cluster and thus it cannot be used in assessing trade off between number of clusters and cluster quality. NMI utilizes an entropy calculation to determine cluster quality and is advantageous as it isn’t affected by differing numbers of clusters. Lastly Rand index is a method of determining similarity between two clusters using actual and predicted labels. Each of these metrics has advantages and disadvantages. It is also possible that a clustering metric specific to this project will be developed but this has not yet been determined. The decision of which metric to use will depend on the outcome of the final phase of the project and will be decided when that phase of the project is reached.



## Plan
1. Baseline Model Development
  * Setting up our development environment (git, cloud resources, etc)
  * Preprocess and clean data (possibly: lowercasing text, stripping special characters, removing stopwords)
  * EDA to identify relevant keywords, hashtags, and potential computed fields
  * Train a sci-kit learn model using one-hot encoding 
  * (Feb 18): First Demo
2. Improved Model Development
  * Research advanced NLP techniques like the Attention metric 
  * Improve one-hot encoded model with new metrics identified in previous step.
  * (March 4): Progress Report Due
3. Final Model Development and Interactive Dashboard
  * Scrape Twitter data
  * Apply the same preprocessing functions developed in second step of Baseline Model Development
  * Geocode the tweets to approximate coordinates from the given town, city, or county. (Probably use Google Geocoding API)
  * Classify all tweets as “relating to” or “not relating to” a real disaster.
  * Develop and publish a Tableau Public dashboard that:
    * Plots the location of all scraped tweets on a Mapbox ma with dots.
    * Colors the dots of tweets “relating to” disasters in red and tweets “not relating to” disasters in blue.
    * Contains a scroll filter to show tweets only from specific hour intervals. (Allows viewers to watch the progression of a disaster as it occured)
  * (April 15): Project demo on April 15
  * (April 22): Interactive dashboard demo


## Architecture
The data obtained from Kaggle contains the contents of the Tweet, an associated keyword, a location, the associated id, and the target for the training set all in a .csv format.  The only fields that are not blank for all data points are the id and the text of the Tweet.  By nature, the Twitter data is messy, with associated links, emojis, and extraneous words needed to be removed in order to extract important information. 

In terms of Python packages, scikit-learn will be used for more standard machine learning techniques and spaCy will be used for the bulk of the Natural Language Processing.  These packages will allow us to easily implement algorithms on our data and train our model.  For the Twitter data scraping, we will need to set up an AWS EC2 instance in order to accomplish this at scale.  Additionally, we will make use of a Sagemaker notebook to train our model in the cloud.  The vast amount of Twitter data we will need to mine in order to create our heat map necessitates this use of the cloud. 

For our visuals, the simulation could be stored in one of two ways: a custom webpage that will make use of HTML and Google maps or a Tableau public dashboard.  The Tableau dashboard is the most likely course of action, as it allows us to easily create and disseminate our simulation.


## Communication
A group Discord has been established in order to ask specific questions and keep up to date with anything that needs to get done with the project.  Additionally, we have set up weekly meetings on Tuesdays and Fridays in order to stay on top of the progress of our project.  The Tuesday meetings are longer and are devoted to planning our next steps and working on anything that needs to get done.  The Friday meetings function as check-ins to ensure that the project timeline remains accurate.  We will also make use of class time to make progress on the project.

## Ethics Checklist

- [x] Can we shut down this software in production if it is behaving badly?
	* Yes, in the situation that the model was producing biased results or if in any way, it appeared that there were concerning errors, the software could 	easily be shut down and pulled from use. 
- [x] Do we test and monitor for model drift to ensure our software remains fair over time?
	* Were we able to deploy our model to production (meaning as a method of real-time disaster detection via Twitter) we would continuously monitor its 		performance in detecting real-world disasters. If we see that accuracy or specificity declines over time, we could augment the original database with new 	  data we obtained ourselves and retrain the model. This is a likely scenario as language changes subtly over time and prior linguistic cues may become less 	     relevant as others become moreso. 
- [x] Do we have a plan to protect and secure user data?
	* All of the data that we have collected contains no information about the user who made the Tweet.  This means that there is no way to trace any specific 	   Tweets back to the user, so there is no risk of user information being leaked. Our data and analyses are being hosted in a private GitHub repository and any 	public visuals will be configured to not expose any of the model’s source data.
