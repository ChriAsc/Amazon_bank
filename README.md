# Prediction of the Amazon share price trend and analysis of a promotional campaign banking via the Python language

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

- differentiated series I order: ADF statistics: -25.37 1 , p-value: 0.0;
- differentiated series 2nd order: ADF statistics: -11.7 50 , p-value; 1.2*10-^21 ;

From these tests, first-order differentiation can be considered, as
further differentiation could result in the loss of features
important features of the series, distorting it.

For the prediction of a time series through its past values, one
you want to test different approaches: first you want to create an ARIMA model
and then you want to add the seasonal component, so as to obtain a
SARIMA model. Generally, an ARIMA model is characterised by three
parameters, namely **p** (Auto Regressive), **q** (Moving Average) and **d** (Integrated), which
which can be calculated using certain tests, such as the ADF test.

Having already performed this test, it can be assumed that 𝑑= 1 , as the p-
value corresponding to the one-time differentiated series is less than 0.05.

In order to evaluate p and q, the autocorrelation
partial and autocorrelation respectively. From the graph of the PACF (Partial Auto Correlation
Function), it can be seen that only the _lag(1)_ is above the level of
significance level, so we set 𝑝= 1. Similarly, from the graph of the ACF
(Auto Correlation Function), it follows that only one term is needed
to remove any autocorrelation in the stationary series, so 𝑞= 1.


### 3.4 ARIMA model

Since the characteristic parameters have been evaluated (𝑝= 1 ,𝑑= 1 ,𝑞= 1 ), we
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

We consider the configuration (1,1,1)x( 1 , 3 , 4 ,5) and can observe in (Figure
15 ) that the prediction follows the trend; in fact, initially the first
_bottom_ , then the curve stabilises, without predicting the final peak.

Subsequently, a different configuration of the
SARIMA model, namely (1,1,1)x( 10 ,3, 7 ,5). In this case (Figure 16 ), we
notice that, due to the MA terms, there is a higher fidelity towards the movements
of the actual curve (more evident peaks). On the other hand, there is a translation
of the declines and rises, even though these follow the course of the curve, which
which is within the confidence interval (95%).

### 3.6 Residuals and evaluation metrics

Having obtained the models, one can proceed with the analysis of residuals. In fact, it is
possible to identify any correlations or if there is information
that could be useful for forecasting. For the first SARIMA model
(Figure 17 ), it can be seen that the residuals stabilise, as is also shown by the
slightly shifted distribution compared to the normal distribution. However, one can
notice from the correlogram that correlations are present between the first residuals.

With regard to the second SARIMA model, one can see a situation very similar to the previous one (Figure 18).
similar to the previous one (Figure 18 ). In contrast, however, to the configuration
(1,1,1)x(1,3,4,5) configuration, in this case the correlogram does not show any particular
anomalies and the distribution is much more similar to a Gaussian.

To measure the goodness of prediction, a number of metrics were used,
namely the MAPE (Mean Absolute Percentage Error), the ME (Mean Error), the
MAE (Mean Absolute Error), the MPE (Mean Percentage Error), the RMSE (Root
Mean Squared Error ), the ACF (autocorrelation of the error for lag( _1_ )), the
correlation between the series and the prediction, the MINMAX error.

Considering the SARIMA(1,1,1)x(1,3,4,5) model, as can be seen in
Figure 19 , it can be seen that the MAPE equals 0.077, the RMSE equals
277.5, while the CORR stands at 0.31.

On the other hand, with regard to SARIMA (1,1,1)x(0,3,7,5), we have MAPE and
RMSE of 0.167 and 532.2 respectively, while the CORR is equal to
0.45 1.

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
