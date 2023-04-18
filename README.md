# Prediction of the Amazon share price trend and analysis of a promotional campaign banking with Python

## 1. Introduction
The following report illustrates the project we carried out in the field of
clustering/classification and time series analysis.

The programming language used throughout the development is
Python, which over the years has established itself as the standard for
data science; this is due to the fact that it offers countless libraries
libraries (such as Pandas, SKLearn and StatsModel) which represent the state of the
of the art and are accessible free of charge.

## 2. Datasets

After a long search for a dataset on which to base the
project, we came to the conclusion that, given the great diversity of the tasks
to be solved, a single dataset would not be sufficient. For this reason
we decided to use two, one for time series analysis and one
for classification/clustering.

The datasets in question are as follows:

https://www.kaggle.com/datasets/varpit94/amazon-stock-data for analysis
time series analysis.

https://www.kaggle.com/datasets/kidoen/bank-customers-data for
clustering and classification.

The first of the two is called " _Amazon Stock Data_ " and is a dataset
containing the stock market performance of Amazon's stocks from 15 May
1997 to the present day.

In detail, the component columns of the dataset are the following seven.

**Date** : Reference date in "yyyy-mm-dd" format.

**Open** : Price of the first transition of the day.

**High** : Maximum price on the reference day.

**Low** : Lowest price on the reference day.

**Close** : Price of the last transition on the day.

**Adj Close** : Closing price adjusted to reflect any actions taken by the corporate.

**Volume** : Number of units traded during the day.

The nature of this dataset makes it perfect for performing time series analyses, as most of the fields are surveys made with
almost constant time intervals (weekends are not present in the days).


The second dataset is called 'Bank Customers Data' and contains a set
of data relating to the customers of a Portuguese bank. These data were
collected during a direct marketing campaign. The purpose behind the
dataset is to be able to understand, based on the profile of a customer, whether the
a client's profile, whether or not the latter would entrust the bank with a long-term deposit.
not.

As in the previous case, the dataset is in .csv file format and the columns within it are as follows.
within it are as follows.

**Age** : Integer field indicating the person's age.

**Job** : String field containing the occupation of the person.

**Marital** : String field indicating the marital status of the person.

**Education** : String containing the level of education attained by the person.

**Default** : Boolean field indicating whether the person's account is in default

**Balance** : Numeric field indicating the customer's bank credit

**Hougsing Loan** : Boolean field indicating whether the person has received a loan for the purchase of a property.

**Loan** : Boolean variable indicating whether or not the person has received a loan of any kind.

**Contact** : String indicating the type of contact details for contacting the person.

**Day** : Day of the contact (survey) with the customer.

**Month** : Month of the contact (survey) with the customer.

**Duration** : Duration of the contact (survey) with the customer (in seconds).

**Campaign** : Number of attempts made to contact the customer for the purposes of the

survey in the current campaign.

**Pdays** : Number of days since the last contact with the customer.

**Previous** : Number of calls made to the customer prior to the current campaign.

**Poutcome** : Result of the last campaign (success or failure).

## 3. Time Series

The first analysis we carried out was on the time series,
based as mentioned above on the dataset " _Amazon Stock Data_ ". The general purpose
general purpose behind these analyses was to be able to observe
trend of these measurement series and make predictions on them after an initial pre-processing phase.
an initial pre-processing phase.

### 3.1 Initial Processing

Initially, using Pandas' default _to_datetime_ method, the
Date' field was converted to datetime format, which is easier to
manipulate. In addition, the time reference was transformed
into the index of the dataset using the _set_index_ function. Subsequently, only the
only the 'Open' column was considered, as the analysis focuses
on the opening price of the share; however, the difference between the
opening and closing price is not sufficiently important to
justify the use of both fields; in Figure 2 it can be seen that
the delta between the curves is limited.

From the analysis of the columns, it can be seen that the opening days of the
stock market are usually five (Monday to Friday), excluding some
holidays (1 January, 4 July, etc.).

Given the large number of observations, we decided to restrict the dataset
to the date range from 2 January 2020 to 24 March 2022; this
choice is due to the fact that our analyses concern the future trend
of the stock market and consequently having data that is too far back risks
pollute the forecasts, also in light of more recent events and the
less traditional economic situation.

Only the reported operations of filtering and
transformation of the data because the starting dataset was already clean and ready to use (the
ready to use (the relative usability coefficient reported by Kaggle is 10).
Figure 4 shows a sample extracted from the "Open" column.


### 3.2 Study of the characteristics of the series

First, we proceeded with the decomposition of the series in order to
understand the development of the three main components: trend, seasonality and
residuals. This made it possible to visualise the isolated characteristics of the
series, both through additive decomposition (Figure 5 ) and through multiplicative decomposition (Figure 6 ).
multiplicative decomposition (Figure 6 ). Concerning the residuals, those related to
additive decomposition are less regular, so this is preferable to the multiplicative one.
the multiplicative one.

Having done the decomposition, one can clearly see both a
tendency towards higher values and a certain seasonality. Focusing on the
trend, a more in-depth study can be carried out by means of a
detrending, so as to visualise the series without the most obvious component.
evident. In fact, thanks to the detrend function provided by the
scipy" library, the "detrended" series (Figure 7 ) shows that, even if they are not
being equal in absolute value, the price in April 2020 and that in
March 2022 indicate a similar economic situation: in fact, in the first
case the downturn was due to the initial impact of the pandemic, whereas in the second case the
second case, the behaviour is attributable to both the war and the rise
of interest rates.


### 3.3 Stationarity analysis and parameter estimation

Since the analysis of a time series is simpler if
this is stationary, it is necessary to proceed with the verification of this property.
In fact, in a stationary series, the values do not depend on time and, therefore,
mean and variance are constant. For this purpose, we use
the Augmented Dickey Fuller test is used, in which the series is considered non-stationary 
(null hypothesis) and the p-value is calculated to confirm or reject this hypothesis:
if the p-value > 0.05 the series is non-stationary, otherwise the null hypothesis is
rejected and the series is asserted to be stationary.

In the case of the integral series, the test shows that the ADF statistics is -
2.343, while the p-value is 0.158, so it can be stated that the
series is non-stationary.

Therefore, a differentiation of the series is performed, of first and
second order. After this, the ADF test is performed on the latter two series
obtained, obtaining:

- differentiated series I order: ADF statistics: -25.371, p-value: 0.0;
- differentiated series 2nd order: ADF statistics: -11.750, p-value; ${ 1.2 \times 10^{-21} }$ ;

From these tests, first-order differentiation can be considered, as
further differentiation could result in the loss of features
important features of the series, distorting it.

For the prediction of a time series through its past values, one
you want to test different approaches: first you want to create an ARIMA model
and then you want to add the seasonal component, so as to obtain a
SARIMA model. Generally, an ARIMA model is characterised by three
parameters, namely **p** (Auto Regressive), **q** (Moving Average) and **d** (Integrated), which
which can be calculated using certain tests, such as the ADF test.

Having already performed this test, it can be assumed that 𝑑=1 , as the p-
value corresponding to the one-time differentiated series is less than 0.05.

In order to evaluate p and q, the autocorrelation
partial and autocorrelation respectively. From the graph of the PACF (Partial Auto Correlation
Function), it can be seen that only the _lag(1)_ is above the level of
significance level, so we set 𝑝= 1. Similarly, from the graph of the ACF
(Auto Correlation Function), it follows that only one term is needed
to remove any autocorrelation in the stationary series, so 𝑞= 1.


### 3.4 ARIMA model

Since the characteristic parameters have been evaluated (𝑝=1,𝑑=1,𝑞=1), we
proceeded to create and train the ARIMA model.

The first prediction was calculated using the _predict()_ method, whose
attribute ( _dynamic_ ) which can be set to _True_ (at each step a prediction is
performed a prediction by adding the value predicted in the previous step,
readjusting the model) or _False_ (sequential prediction from the previous true value, whereby the prediction
previous true value, whereby the prediction is shifted by one step). The results of the
prediction were not satisfactory, as can also be seen with
_dynamic = False_ (Figure 12 ).

A _cross validation_ was also performed, dividing the dataset into training
set (90%) and testing set (10%), in order to verify the goodness of prediction;
However, with a confidence interval of 95%, the prediction does not reflect the actual trend (Figure 13).
actual trend (Figure 13 ).

A study of the residuals was carried out (Figure 14 ), in order to note any irregularities or patterns.
any irregularities or patterns. It can be seen that these are contained and the
average is close to 0, with a low variance, outlining a distribution very similar to the normal one.
distribution very similar to the normal distribution.


### 3.5 SARIMA model

An extension of the ARIMA model is the SARIMA model, which incorporates
the ARIMA and also includes a seasonal component. In fact, in many
cases it is possible to detect a certain seasonality. Through the graphs
of autocorrelation and decomposition, one can see a slight
seasonality every 5 lag (equivalent to 5 days). Starting from the same
parameters (p, d, q) of the ARIMA model, four more parameters are added
(P, D, Q, m): the parameter **m** indicates the seasonality of the series, the rest refer to the seasonal component.
refer to the seasonal component. The construction and training of the
model is carried out as before, implementing the same
division of the dataset into training and testing set.

We consider the configuration (1,1,1)x(1,3,4,5) and can observe in (Figure
15) that the prediction follows the trend; in fact, initially the first
_bottom_ , then the curve stabilises, without predicting the final peak.

Subsequently, a different configuration of the
SARIMA model, namely (1,1,1)x(10,3,7,5). In this case (Figure 16), we
notice that, due to the MA terms, there is a higher fidelity towards the movements
of the actual curve (more evident peaks). On the other hand, there is a translation
of the declines and rises, even though these follow the course of the curve, which
which is within the confidence interval (95%).

### 3.6 Residuals and evaluation metrics

Having obtained the models, one can proceed with the analysis of residuals. In fact, it is
possible to identify any correlations or if there is information
that could be useful for forecasting. For the first SARIMA model
(Figure 17), it can be seen that the residuals stabilise, as is also shown by the
slightly shifted distribution compared to the normal distribution. However, one can
notice from the correlogram that correlations are present between the first residuals.

With regard to the second SARIMA model, one can see a situation very similar to the previous one (Figure 18).
similar to the previous one (Figure 18). In contrast, however, to the configuration
(1,1,1)x(1,3,4,5) configuration, in this case the correlogram does not show any particular
anomalies and the distribution is much more similar to a Gaussian.

To measure the goodness of prediction, a number of metrics were used,
namely the MAPE (Mean Absolute Percentage Error), the ME (Mean Error), the
MAE (Mean Absolute Error), the MPE (Mean Percentage Error), the RMSE (Root
Mean Squared Error), the ACF (autocorrelation of the error for lag( _1_ )), the
correlation between the series and the prediction, the MINMAX error.

Considering the SARIMA(1,1,1)x(1,3,4,5) model, as can be seen in
Figure 19, it can be seen that the MAPE equals 0.077, the RMSE equals
277.5, while the CORR stands at 0.31.

On the other hand, with regard to SARIMA (1,1,1)x(0,3,7,5), we have MAPE and
RMSE of 0.167 and 532.2 respectively, while the CORR is equal to
0.451.

Thus, from the evaluation metrics, it can be asserted that the first model
model is more accurate in general, while in the second model there is a
higher correlation between the original series and the forecast.

In conclusion, from the analysis of this time series, it was obtained that, as
supposable, predicting the performance of a stock on the stock exchange is
highly complicated and that the resulting forecast is not very accurate.
accurate. Indeed, especially in the last year, first because of the pandemic
then because of the war, there have been upheavals in the global
global economic scenario. As a result, there is greater volatility and, due to the
presence of exogenous factors, one might think of refining an analysis of
this type of analysis with a SARIMAX-type model. For example, one could
consider other economic indices, such as interest rates, gold, oil,
inflation, the VIX.

## 4. Classification and Clustering
### 4.1 Descriptive analysis

The first step in this phase of the project involved carrying out
descriptive analyses, i.e. all those analyses designed to illustrate the nature of the
dataset, showing its composition and highlighting any correlations.

We therefore focused initially on this phase of Data
Understanding, going to examine the dataset from various points of view. Of
below are some of the analyses carried out in order to have a better view of the dataset.
better view of the dataset.

The first was an analysis concerning the average duration of the
advertising calls on a time basis (division by months). The 'duration' column
represents a crucial piece of information, as in the
vast majority of cases, if a customer stays on the phone listening
listening to an advertising message implies that there is some interest on their part in the service offered.
towards the service offered. The average duration calculation showed that in the
months prior to winter there is a slight increase in the latter
value, peaking in December; this could be explained by considering that the period around
explained by the fact that the period around Christmas has always been
characterised by an increase in average expenditure (Figure 21).

A second group of analyses concerned the general characteristics of the
people affected by the advertising campaign. There are pie-
charts representing different customer characteristics, such as
sector of work, age group, marital status and level of education. This
characterisation of the records in the dataset allows a clearer view of the
a clearer view of the people covered by the survey, and makes it possible to
understand which fields are the most relevant in order to carry out the
classification and segmentation.

A final example that may be useful for understanding the dataset
concerns the analysis carried out with regard to the amount of customer budgets. Of
fact, the focus at this stage was to observe the amount of average bank credit as the
average bank credit as certain characteristics of the persons
concerned. The most surprising fact that emerged from this analysis concerned the
people of an older age. In fact, it seems that senior citizens are
those with a higher average balance than all the other categories;
this can be seen from the bar graphs shown, where it is clear that people
between 71 and 80 years of age are those with the richest bank accounts (Figure 26).
richest bank accounts (Figure 26 ); this behaviour is confirmed by looking at the average balance
average balance per occupation (Figure 27), where it can be seen that the
people who have now retired from work (often pensioners) have a higher average
higher average than all other classes.

### 4.2 Classification

The main purpose of classification is to assign a category
to a set of input elements. This is a task that, depending on the
techniques used, may fall into supervised or unsupervised learning.

The algorithms (given below) used all fall into the category of
supervised learning:

- Logistic Regression
- Decision Tree
- Support Vector Classifier (SVC)
- Random Forest
- AdaBoost
- Gradient Boosting
- Classification resulting from Linear Discriminative Analysis

Our desired classification concerns the _term_deposit_ attribute in the dataset.
present in the dataset; in fact, given the profile of a client (feature values)
the objective was to understand whether the latter would be able to entrust the bank with a
bank a deposit in the long term. Consequently, the task performed was
a binary classification task between the label "Yes" (the customer could
make a long-term deposit) and the label "No" (the customer will not
entrust the bank with a long-term deposit).

As mentioned earlier, the algorithms used are all supervised
supervised learning, consequently the quality of the final models depends
strongly on the composition and goodness of the training dataset. In this
in this regard, a problem that became apparent from the earliest stages was that of class-imbalance.
was that of class imbalance; this means that in the starting dataset there was a very strong
there was a very strong imbalance between the class "Yes" and the class "No".
In most cases, this can lead to final models
characterised by the dominance of a bias that would tend the
classification towards the majority class. For example, if we were to have
a dataset composed of 999 class 1 elements and only one element of
class 2, the classifier produced by training on that dataset would indistinctly label
indistinctly all the inputs as belonging to class 1.

In our case, model training on the starting dataset did not produce such extreme
such extreme results; on the contrary, the latter were from the outset
quite satisfactory, showing a general accuracy of around 90%.

The problem related to class-imbalance was more apparent after calculating
the precision and recall metrics by individual class using Logistic
Regression. In fact, doing so showed that the precision value for the
class 1 (Si) was quite disappointing, hovering around 58%; this
indicates that when the classifier labels with class 1, it is wrong almost
half of the time. However, the most serious metric was recall, at 17%; this
means that out of the total number of 'interesting' customers, only 17% were
correctly identified as such. In contrast, both metrics
relative to the majority class were satisfactory and well over
above 90%.

### 4.3 Approach to class-imbalance

In order to have a more balanced dataset, it is necessary to adopt techniques
to have a better balance between classes. The first
approach we have adopted is that of _metacost_.

**Metacost**

This is a meta-algorithm that functions as an extension to existing approaches
existing approaches and its objective is to relabel the dataset so that
this becomes balanced again. The first step in applying the meta
cost consists of generating _m_ training sets from the initial one with the
bootstrap technique (random extraction with reinsertion). This is done,
_m_ models are trained using the chosen algorithm on the _m_ datasets
generated; each _x_ element of the initial dataset is labelled by all _m_
models, generating _m_ generically different predictions. The final
final probabilities for each class _P(j|x)_ as the sum of the number of times that
_x_ was labelled as belonging to class _j_. This is done by calculating the
cost of reassigning element _x_ to class _i_ (using the formula
below for each class _i_ ) the new label of _x_ is found.

That is, _x_ will be relabelled as class _i_ , where the latter minimises the
summation of the products of the costs of confounding i with each of the
other classes by the probabilities of all the other classes. Having relabelled all the
elements of the starting dataset, the metacost returns the model _M_ obtained
by applying the algorithm chosen at the start to the relabelled dataset.

Having chosen an implementation of the metacost, error costs
equal to 4 in the case of class 1 (Yes) and equal to 1 in the case of class 0 (No); the value
4 causes the error in the case of class 1 to be judged heavier in the
training.

The results in terms of accuracy, precision and recall are shown below and,
as can be seen, the improvements are considerable.

**Manual balancing**

Another approach taken is manual _sampling_. For
first, an undersampling of the majority class was performed,
respecting the distribution of the dataset, so as to decrease it to approximately one
third of the original size. Next, an
oversampling was applied to the minority class, so that the two classes had the
same number of elements.

At this point, the models were trained and the same metrics were calculated as before.
same metrics as previously seen. Generally, the results are
sufficiently good, but the best performance is obtained with the Random
Forest (Figure 31 ): accuracy settles at 0.95, precision and recall are
between 0.92 and 0.98.

The confusion matrices (Figure 32 ) show that the models trained and
tested have more or less similar behaviour, although in general the
classification trees (thus also the Random Forest) predict the
better. In the confusion matrix, we represent on the rows the
current values and on the columns the predicted values.

Another useful method for evaluating the models obtained is the ROC curves
curves (Figure 33 ). In general, the ROC curve shows the performance of a model
classification considering all possible classification thresholds. This
curve represents two parameters: _True Positive Rate_ and _False Positive Rate_.

Subsequently, the presence of correlation between the
prediction models (Figure 34 ). In detail, the trees are correlated with the Random
Forest, while there is some correlation between the logistic regression model
logistic regression model, the Gradient Boosting classifier and LDA.

### 4.4 Clustering

The second type of task tackled in this project was the
clustering. This indicates the operation of dividing a set of elements
provided as input into subgroups that have certain characteristics in common.
This division into clusters takes place in the feature space. The latter
is defined as a space having as dimensions the characteristics that
distinguish the elements of our dataset (features). In this type of
In this type of task, the choice of these features is of fundamental importance, i.e. on which of them to base the division.
which of them to base the division on.

There are various types of algorithms designed to perform clustering, but
the one chosen is K-Means. The latter falls into the category of
prototype-based, i.e. those that are based on moving the centroids
of the cluster (centre points) and define the membership of an element in
a cluster based on the nearest centroid.

In the case of K-means there is an initialisation of the centroids, and a
a first problem arises, namely how many centroids (i.e. clusters)
must be predicted. The easiest method to find an answer is called the
"Elbow Method'. This method consists of applying the k-means algorithm
by increasing the number of clusters used each time. Then, for each
of the segmentations obtained, the average coefficient of
distortion in the clusters obtained; one can think of this value as the
average distance of a point from the centre of the assigned cluster.

Plotting the course of the distortion as a function of the number
(increasing) number of clusters used, one will notice both that the average distance is initially
very high (few clusters) and that it has, in the early stages, a strong
tendency to decrease. When a sufficient number of
clusters the distortion stabilises, this is because the segments obtained are
such a small size that subdividing them further does not lead to any
substantial changes in the distortion coefficient. Having obtained
this curve, the suitable number of clusters is identified by the point at which
the curve changes from a steep descent to a more stable situation.
of greater stability.

As can be seen, the suitable number of clusters for our dataset
appears to be between 3 and 7 (red area in Figure 35).

A second check that can be carried out, in order to identify the most suitable number of
suitable number of sub-clusters is the one concerning the _coefficient silhouette_ (Figure 36).
This coefficient varies between 1 and -1. Numbers close to 1 indicate that the
elements are closer to the centre of their cluster than to the centres of the
surrounding clusters (neighbouring clusters). Scrolling again through the various
various cluster numbers calculating each time, this coefficient will give us
a second confirmation as to how many clusters to use.

The analysis of the silhouette coefficient indicates that, as the number of
clusters, there is an almost linear worsening of this value. This means
that we should choose as few segments as possible.
Considering also the analysis using the elbow curve, we chose a
number of clusters **no more than 4.

Having carried out several tests with different numbers of clusters, we obtain a
discrete segmentation of the dataset with 4 clusters (Figure 37 ).

As can be seen, the quality of the segmentation is not optimal and the
silhouette lies between 0.5 and 0.6. In particular, the
feature pairs 'age' - 'balance' and 'balance' - 'duration' were considered, since with the
other categorical features, the resulting clustering presented clusters
clustered for the values of the latter. With this configuration, one can
identify four clusters corresponding to four income bands.
Firstly, cluster 2 is representative of the poor bracket (around 0,
also including lower values), cluster 4 represents the middle- to low
low bracket (close to € 20000), cluster 3 corresponds to the richest bracket,
while cluster 1 represents the upper-middle class.

As the dataset contains a large amount of items, it was decided to
take a smaller sample, respecting the original distribution.
Therefore, a reduction of the dataset was implemented and the K-
means on 150 points. Thus, the same pairs of features were considered as previously
features as seen before and four clusters were obtained again,
with the resulting silhouette being approximately 0.63. As can be seen in
Figure 38 , the clusters identified define the same four groupings
previously described. In this case, there is a sharper division
of the clusters and it can be seen that cluster 2, i.e. the one representing the
poorest bracket, is much more compact than the other three.

Having considered three features with three different scales, a
normalisation was carried out to mitigate the unbalancing effect of the weight given to the
values of the features. Thus, we verified the presence of differences between
the clusters before and after normalisation on the 150 features. In Figure 39,
it can be seen that the clusters are very similar to those previously
identified. In fact, cluster 2, i.e. the one representing the poorest
poorer, unlike without normalisation, is more
distant from cluster 4, which identifies the lower-middle band. On the other hand, as
with regard to the other clusters, there is less separation, as the
proximity between cluster 1 and cluster 3 is due to the presence of more
number of elements following normalisation. Furthermore, unlike
seen previously, the cluster representing the highest band
appears to be more compact. In the same image, an
interesting situation regarding the relationship between "balance" and "duration";
In fact, the middle (medium and low) clusters have a greater interest in the
campaign, as evidenced by a longer stay on the telephone
telephone, compared to the extremes, since in most cases the richer
already have long-term deposits and the poorest have no resources.


To conclude, we would like to draw attention to the composition of the dataset
chosen. The characteristics of the latter do not favour the execution of an
unsupervised task, such as clustering, as, in the feature space, there is a very dense distribution of samples.
features, there is a very dense distribution of samples. This
observation is demonstrated by the fact that the execution of density
based algorithms, such as DBSCAN, based only on the distance separating the
elements, clusters all the samples into a single cluster.

#### _Disclaimer: in-text images refer to 'Relazione_ClassificationTimeSeries.pdf' file_
