# Technical

## Consider the outliers

There were no outliers in terms of data that were siginificantly different than other examples, but we did have to be aware of potential mislabeled data that may have affected our final prediction accuracy.  We decided to keep them in, as it was occasionally difficult to determine if some examples were misclassified and all of the removals would have to be done by hand.

## Look at examples

We were able to look at the breakdown of false positives and negatives (as well as true positives and negatives) in order to make sure our model was not skewed to be more accurate for one of the classes.  We did not notice any gap in misclassification rates between the classes that could skew these results.

## Check for consistency over time

We have not begun the last phase of our project yet, so we don't have data that is spread out over time.  However, this practice was applied in a different sense, as we had to ensure that our model performed about the same regardless of where the training data was split.

## Slice your data

Splitting our data along the keyword feature allowed us to visualize if any of the keywords were being misclassified at a higher rate than others.  It turned out the there was not a tremendous amount of variance in the AUC across the different keywords.

# Process

## Confirm experiment and data collection setup

We manually reviewed many rows from the Kaggle dataset to ensure they were properly tagged and were materially different from what we were expecting. The tweets were not dated, timestamped, or geotagged, so we could not verify a representative sample that way. The point made about reproducibility in the next summary applies here as well - the performance of the model on independently-collected data should give us some idea of how representative the source data was.

## Check for reproducibility

Since the third part of the project will require using our model on independently collected Twitter data, we will get some idea of the reproducibility of the model based on our Phase 3 accuracy.

## Exploratory analysis benefits from end-to-end iteration

We have not had the chance to repeat or rethink any aspects of our Exploratory Data Analysis. Due to complications implementing our model on the Temple GPU servers, we haven’t had as much spare time as we were hoping for in Phase 2.

## Check for consistency with past measurements

There is no “other” dataset of disaster tweets for us to directly compare to. Any consistency checks will be done with our independently collected data. That said, we can directly compare the performance of our model to other submissions on Kaggle. We are still falling a bit short of our goal of an f1-score around 0.85. In the future, we might compare our results to other teams on Kaggle using Simple Transformers or BERT.

# Mindset

## Data analysis starts with questions, not data or a technique

In starting out with the model building, we spent time examining the data before building any model in order to ensure the right questions were being addressed. Also, phase 2 of the project has evolved around asking different questions about what tools such as Optuna or GPU processing are most effective for addressing the specific Twitter dataset that is being used. As various problems have arisen, the focus has shifted to accommodate and address them. 

## Be both skeptic and champion

This point will come into practice as the model built during the earlier parts of the project is applied to the real world disaster data that we collected. We’ll assess whether model performance is maintained as it is tested on a new dataset. Also, the model’s usefulness in an applied setting will be examined and the results of that can be celebrated.

## Share with peers first, external consumers second

Throughout the project, the team has maintained regular weekly meetings to check in with each other and update each other on progress while also sharing ideas in a collaborative environment. 

## Expect and accept ignorance and mistakes

Each team member has worked on different parts of the project and as different problems have arisen, these parts have been passed off to each other in an effort to encourage one another and work together to bring out everyone’s strengths. Also, during weekly team meetings, the different issues that each of us have run into are discussed with  group brainstorming for solutions.
