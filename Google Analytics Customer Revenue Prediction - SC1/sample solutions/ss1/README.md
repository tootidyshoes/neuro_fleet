# Google-Analytics-Customer-Revenue-Prediction
Predict how much GStore customers will spend.
The project is built to participate in the competition on Kaggle.com

## Description
The 80/20 rule has proven true for many businesses–only a small percentage of customers produce most of the revenue. As such, marketing teams are challenged to make appropriate investments in promotional strategies.

In this project, we are going to analyze a Google Merchandise Store (also known as GStore) customer dataset to predict revenue per customer. Hopefully, the outcome will be more actionable operational changes and a better use of marketing budgets for those companies who choose to use data analysis on top of GA data.

## Evaluation
The prediction is evaluated on the root mean squared error. RMSE is defined as:

<a href="https://www.codecogs.com/eqnedit.php?latex=\textrm{RMSE}&space;=&space;\sqrt{\frac{1}{n}&space;\sum_{i=1}^{n}&space;(y_i&space;-&space;\hat{y}_i)^2}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\textrm{RMSE}&space;=&space;\sqrt{\frac{1}{n}&space;\sum_{i=1}^{n}&space;(y_i&space;-&space;\hat{y}_i)^2}" title="\textrm{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}" /></a>

where y hat is the natural log of the predicted revenue for a customer and y is the natural log of the actual summed revenue value plus one.

## Data description

### Source

- You will need to download **train.csv** and **test.csv**. These contain the data necessary to make predictions for each fullVisitorId listed in sample_submission.csv. All information below pertains to the data in CSV format.

Download here (https://www.kaggle.com/c/10038/download-all)



### Format
Both **train.csv** and **test.csv** contain the columns listed under **Data Fields**. Each row in the dataset is one visit to the store. 

### What am I predicting
We are predicting the natural log of the sum of all transactions per user. For every user in the test set, the target is:

<a href="http://www.codecogs.com/eqnedit.php?latex=y_{user}&space;=&space;\sum_{i=1}^{n}&space;transaction_{user_i}" target="_blank"><img src="http://latex.codecogs.com/gif.latex?y_{user}&space;=&space;\sum_{i=1}^{n}&space;transaction_{user_i}" title="y_{user} = \sum_{i=1}^{n} transaction_{user_i}" /></a>

<a href="http://www.codecogs.com/eqnedit.php?latex=target_{user}&space;=&space;\ln({y_{user}&plus;1})" target="_blank"><img src="http://latex.codecogs.com/gif.latex?target_{user}&space;=&space;\ln({y_{user}&plus;1})" title="target_{user} = \ln({y_{user}+1})" /></a>

### File Descriptions
**train.csv** - the training set

**test.csv** - the test set

**sampleSubmission.csv** - a sample submission file in the correct format. Contains all fullVisitorIds in **test.csv**.

### Data Fields
**fullVisitorId** - A unique identifier for each user of the Google Merchandise Store.

**channelGrouping** - The channel via which the user came to the Store.

**date** - The date on which the user visited the Store.

**device** - The specifications for the device used to access the Store.

**geoNetwork** - This section contains information about the geography of the user.

**sessionId** - A unique identifier for this visit to the store.

**socialEngagementType** - Engagement type, either "Socially Engaged" or "Not Socially Engaged".

**totals** - This section contains aggregate values across the session.

**trafficSource** - This section contains information about the Traffic Source from which the session originated.

**visitId** - An identifier for this session. This is part of the value usually stored as the utmb cookie. This is only unique to the user. For a completely unique ID, you should use a combination of fullVisitorId and visitId.

**visitNumber** - The session number for this user. If this is the first session, then this is set to 1.

**visitStartTime** - The timestamp (expressed as POSIX time).

### Removed Data Fields
Some fields were censored to remove target leakage. The major censored fields are listed below.

**hits** - This row and nested fields are populated for any and all types of hits. Provides a record of all page visits.

**customDimensions** - This section contains any user-level or session-level custom dimensions that are set for a session. This is a repeated field and has an entry for each dimension that is set.

**totals** - Multiple sub-columns were removed from the totals field.
