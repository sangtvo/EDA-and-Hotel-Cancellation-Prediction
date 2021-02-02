# Capstone Project: EDA &amp; Hotel Cancellation Prediction
> This study analyzes hotel data from the southern part of Portugal to identify why guests are cancelling their reservations and the potential indicators that are causing them to do so. The original data is derived from Nuno Antonio, Ana Almeida, and Luis Nunes who are researchers from Lisbon University Institute and later cleaned/uploaded on [Kaggle](https://www.kaggle.com/jessemostipak/hotel-booking-demand). Python is the choice of program and will be using (1) KNN (2) Logistic Regression and (3) Random Forest techniques.

Table of Contents
---
1. [General Information](#general-information)
2. [Summary](#summary)
3. [Tech Stack](#tech-stack)
4. [Data Preprocessing/Cleaning](#data-preprocessingcleaning)
5. [Exploratory Data Analysis](#exploratory-data-analysis)
6. [K-Nearest Neighbors](#k-nearest-neighbors-knn)
7. [Logistic Regression](#logistic-regression)
8. [Random Forest](#random-forest)
9. [Solution](#solution)
10. [Key Takeaways](#key-takeaways)

<a name="https://github.com/sangtvo/EDA-and-Hotel-Cancellation-Prediction#general-information"/>
<a name="https://github.com/sangtvo/EDA-and-Hotel-Cancellation-Prediction#summary"/>
<a name="https://github.com/sangtvo/EDA-and-Hotel-Cancellation-Prediction#tech-stack"/>
<a name="https://github.com/sangtvo/EDA-and-Hotel-Cancellation-Prediction#data-preprocessingcleaning"/>
<a name="https://github.com/sangtvo/EDA-and-Hotel-Cancellation-Prediction#exploratory-data-analysis"/>
<a name="https://github.com/sangtvo/EDA-and-Hotel-Cancellation-Prediction#k-nearest-neighbors-knn"/>
<a name="https://github.com/sangtvo/EDA-and-Hotel-Cancellation-Prediction#logistic-regression"/>
<a name="https://github.com/sangtvo/EDA-and-Hotel-Cancellation-Prediction#random-forest"/>
<a name="https://github.com/sangtvo/EDA-and-Hotel-Cancellation-Prediction#solution"/>
<a name="https://github.com/sangtvo/EDA-and-Hotel-Cancellation-Prediction#key-takeaways"/>

General Information
---
The capstone project is part of a graduate course in order to graduate at Western Governor's University. A completion of a capstone prospectus, executive summary, and a power point presentation is required to graduate, but will not be uploaded to my repository. 

**To expand the project even further (originally binary logistic regression), KNN and random forest analysis are added.**

Summary
---
The winning model is the **random forest** algorithm with an overall accuracy of 84.50% and precision of 87.43%. This means that the model will correctly predict hotel cancellation 80.21% of the time. In order for hotel management company to reduce their current hotel cancellation rate of ~37%, management should focus on requiring deposits because 80% of the data require no deposits. This can be mitigated if hotels require fees for cancellation or mandatory deposits. When hotels have stricter cancellation policies, guests are less inclined to cancel their reservation and hotel revenue will increase. In addition, lead time is another factor to be targeted. Guests who hold reservations for long periods of time are more likley to cancel. If management offers special offers or larger discount for on-site services when booking in advance, guests are less likely to cancel. 

Tech Stack
---
* Python
* VS Code
* Jupyter Notebook

Data Preprocessing/Cleaning
---
#### Irrelevant:
Drop status because "is_cancelled" feature already determines whether the guest canceled or not as well as dates.
```python
df=df.drop(['reservation_status', 'reservation_status_date'], axis=1)
```

Dropped outlier.
```python
df=df[df['adr'] < 5000]
```

ADR w/ Outlier | ADR w/o Outlier
:-------------------------:|:-------------------------:
![ADRo](https://github.com/sangtvo/EDA-and-Hotel-Cancellation-Prediction/blob/main/images/adr_outlier.png?raw=true) | ![ADRnoo](https://github.com/sangtvo/EDA-and-Hotel-Cancellation-Prediction/blob/main/images/adr_nooutlier.png?raw=true)

Drop more variables after looking at correlation matrix and what variable is relevant to the target variable, ***is_canceled***.
```python
df=df.drop(columns=['reserved_room_type', 'assigned_room_type', 'agent', 'country'])
```

#### Recoding: 
Recode data types.
```python
df['children']=df['children'].astype('int64')
df['country']=df['country'].astype('str')
```

Label Encoding method to convert categorical variables into numeric for machine learning algorithms.
```python
le = preprocessing.LabelEncoder()
df['hotel']=le.fit_transform(df['hotel'])
df['arrival_date_month']=le.fit_transform(df['arrival_date_month'])
df['meal']=le.fit_transform(df['meal'])
df['country']=le.fit_transform(df['country'])
df['market_segment']=le.fit_transform(df['market_segment'])
df['distribution_channel']=le.fit_transform(df['distribution_channel'])
df['reserved_room_type']=le.fit_transform(df['reserved_room_type'])
df['assigned_room_type']=le.fit_transform(df['assigned_room_type'])
df['deposit_type']=le.fit_transform(df['deposit_type'])
df['customer_type']=le.fit_transform(df['customer_type'])
```

#### Missing Data:
Checking for missing data and calculating percentages of the missing data per column.
```python
round(100*(df.isnull().sum()/len(df.index)),2)
```
```
hotel                              0.00
is_canceled                        0.00
lead_time                          0.00
arrival_date_year                  0.00
arrival_date_month                 0.00
arrival_date_week_number           0.00
arrival_date_day_of_month          0.00
stays_in_weekend_nights            0.00
stays_in_week_nights               0.00
adults                             0.00
children                           0.00
babies                             0.00
meal                               0.00
country                            0.41
market_segment                     0.00
distribution_channel               0.00
is_repeated_guest                  0.00
previous_cancellations             0.00
previous_bookings_not_canceled     0.00
reserved_room_type                 0.00
assigned_room_type                 0.00
booking_changes                    0.00
deposit_type                       0.00
agent                             13.69
company                           94.31
days_in_waiting_list               0.00
customer_type                      0.00
adr                                0.00
required_car_parking_spaces        0.00
total_of_special_requests          0.00
reservation_status                 0.00
reservation_status_date            0.00
dtype: float64
```

Fill in missing data with zeroes.
```python
df['children']=df['children'].fillna(0)
df['agent']=df['agent'].fillna(0)
```

Fill value that appears most often in the country column.
```python
df['country']=df['country'].fillna(df['country'].mode().index[0])
```

Dropping variable because it has 94.31% missing data.
```python
df=df.drop(['company'], axis=1)
```

Separate the target variable, ***is_canceled***.
```python
X = df.drop(columns=['is_canceled'])
y = df['is_canceled']
```

Split the data for modeling (70% train, 30% test)
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=123)
```

Standardizing the data for implementable machine learning algorithms.
```python
scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
```

**For the full notebook, please check out Customer Churn Analysis.Rmd or html file in the "code" folder.**

Exploratory Data Analysis
---
<table>
  <tr><th colspan=2>Univariate & Bivariate Analysis</th></tr>
  <tr><td><img src="https://github.com/sangtvo/EDA-and-Hotel-Cancellation-Prediction/blob/main/images/c_deposit_type.png?raw=true"> </td>
  <td><img src="https://github.com/sangtvo/EDA-and-Hotel-Cancellation-Prediction/blob/main/images/c_hoteltype.png?raw=true"> </td></tr>
  <tr><td> <img src="https://github.com/sangtvo/EDA-and-Hotel-Cancellation-Prediction/blob/main/images/c_repeatedguests.png?raw=true"> </td><td> <img src="https://github.com/sangtvo/EDA-and-Hotel-Cancellation-Prediction/blob/main/images/c_year.png?raw=true"> </td></tr>
  <tr><td> <img src="https://github.com/sangtvo/EDA-and-Hotel-Cancellation-Prediction/blob/main/images/hotel_type.png?raw=true"> </td><td> <img src="https://github.com/sangtvo/EDA-and-Hotel-Cancellation-Prediction/blob/main/images/c_specialreq.png?raw=true"></td></tr>
</table>

* There is a large amount of guest reservations that do not require deposits and a little over 33% of those guests cancel.
* City hotel guests are more common in Portugal compared to resort hotel guests and 3x higher cancellation rate than resort hotels.
* Majority of the guests are possibly first-time visitors since they are not repeated guests.
* The most visited year is 2016, which is more than double the previous year. 
* Roughly 50% of the reservations with no special requests are cancelled. 

![adr](https://github.com/sangtvo/EDA-and-Hotel-Cancellation-Prediction/blob/main/images/c_month_adr.png?raw=true)
* August has the most booked reservations which is most likely Portugal's tourist peak season.
* The ***adr*** (average daily rate) for the summer months are higher and are less likely to cancel compared to other months.
  * This can be partly because of vacation and families have already requested time off from work/school.

![month](https://github.com/sangtvo/EDA-and-Hotel-Cancellation-Prediction/blob/main/images/c_month.png?raw=true)
* It seems that late fall and winter seasons, almost half of the reservations will cancel. 
* Even though late spring and summer have higher bookings, it is more common that more than half of the reservations will cancel.
  * This can be partly due to change of plans, scheduling conflicts, or even ilnesses during these times.

![top10](https://github.com/sangtvo/EDA-and-Hotel-Cancellation-Prediction/blob/main/images/top10.png?raw=true)
* A large portion of the bookings are Portugeuse followed by British and French citizens. 

![market_s](https://github.com/sangtvo/EDA-and-Hotel-Cancellation-Prediction/blob/main/images/market_segment.png?raw=true)
* At least 50% of the bookings are made through an ***Online TA*** (online travel agency) and roughly 25% are ***Offline TA/TO*** (offline travel agents and tour operators).

![m_seg_wknight](https://github.com/sangtvo/EDA-and-Hotel-Cancellation-Prediction/blob/main/images/segment_wknight_hotel.png?raw=true)
* Majority of the market segments are heavily skewed and not very many guests stay no more than 5 days. 
* Week night stays are more prominent for resort hotels than city hotels.
  * This can be possible due to the fact that many resort hotels have package deals when staying for longer days. 

![m_seg_wknd](https://github.com/sangtvo/EDA-and-Hotel-Cancellation-Prediction/blob/main/images/segment_wknd_hotel.png?raw=true)
* In most market segments, guests stay for at least 1 weekend night.
* Aviation segment do not book resort hotels, which makes sense since it is very likely they need a short stay for business.

![lead](https://github.com/sangtvo/EDA-and-Hotel-Cancellation-Prediction/blob/main/images/leadtime.png?raw=true)
* The lead time is the number of days that elapsed between the entering date of the booking into the system and the arrival date.
  * Roughly after 50 days, bookings are more likely to be canceled.

#### Correlation Matrix:
![Corr](https://github.com/sangtvo/EDA-and-Hotel-Cancellation-Prediction/blob/main/images/corr.png?raw=true)
* Removed some high correlated features (>0.80) and that are irrelevant to the study: reserved & assigned room type.

K-Nearest Neighbors (KNN)
---
K-Nearest Neighbors is a simple supervised machine learning algorithm that takes a data point and looks at the "k" closest labeled data point. This unassigned data point is assigned with the label that is the majority of the "k" closest data points.

Running the KNN model.
```python
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)
y_pred_knn = knn_model.predict(X_test)
```

Confusion matrix.
```python
print(confusion_matrix(y_test, y_pred_knn))
print(classification_report(y_test, y_pred_knn))
```
```
[[19969  2722]
 [ 3926  9200]]
              precision    recall  f1-score   support

           0       0.84      0.88      0.86     22691
           1       0.77      0.70      0.73     13126

    accuracy                           0.81     35817
   macro avg       0.80      0.79      0.80     35817
weighted avg       0.81      0.81      0.81     35817
```
```python
precision_knn =  precision_score(y_test, y_pred_knn)
acc_knn = accuracy_score(y_test, y_pred_knn)
print('Precision: ',round(precision_knn * 100,4), '%')
print('Accuracy: ',round(acc_knn * 100,4), '%')
```
```
Precision:  77.1683 %
Accuracy:  81.439 %
```
* Precision is the ratio of correctly predicted observations to the total positive predicted observations.
  * TP / (TP + FP)
* Accuracy is the ratio of correct predictions to the total predictions. 
  * (TP + TN) / (TP + FP + FN + TN)

The goal of this study is to have at least 80% accuracy and the KNN algorithm meets this criteria. However, let's take a look at other algorithms. 

Logistic Regression
---
(Binary) logistic regression is a common classification algorithm when the categorical response have only two possible outcomes, in this case, cancelled or not cancelled. This involves regressing the predictor variables on a binary outcome using a binomial link function.

Run the logistic regression model.
```python
lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)
```

Confusion matrix.
```python
print(confusion_matrix(y_test, y_pred_lr))
print(classification_report(y_test, y_pred_lr))
```
```
[[21536  1155]
 [ 6019  7107]]
              precision    recall  f1-score   support

           0       0.78      0.95      0.86     22691
           1       0.86      0.54      0.66     13126

    accuracy                           0.80     35817
   macro avg       0.82      0.75      0.76     35817
weighted avg       0.81      0.80      0.79     35817
```
```python
precision_lr =  precision_score(y_test, y_pred_lr)
acc_lr = accuracy_score(y_test, y_pred_lr)
print('Precision: ',round(precision_lr * 100,4), '%')
print('Accuracy: ',round(acc_lr * 100,4), '%')
```
```
Precision:  86.0203 %
Accuracy:  79.9704 %
```

This model performs slightly worse than KNN algorithms where accuracy is roughly ~2% less. However, the precision is much higher for logistic regression, about ~9% more. This means that the logistic regression model can correctly predict the positives 86.02% of the time. 

![LR_impfeat](https://github.com/sangtvo/EDA-and-Hotel-Cancellation-Prediction/blob/main/images/lr_featimp.png?raw=true)
* Based on the logistic regression model, the top 5 important predictors that cause guests to cancel their reservations or not are: car parking spaces, guests who cancelled before or not, hotel deposits, and market segment. 

Random Forest
---
Random forest is a supervised machine learning algorithm that creates multiple decision trees that are randomized. It is much more robust and accurate than decision trees.

Run the random forest model.
```python
rf_model = RandomForestClassifier(min_samples_leaf=10, min_samples_split=10, n_estimators=100)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
```

Confusion matrix.
```python
print(confusion_matrix(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))
```
```
[[21400  1291]
 [ 4281  8845]]
              precision    recall  f1-score   support

           0       0.83      0.94      0.88     22691
           1       0.87      0.67      0.76     13126

    accuracy                           0.84     35817
   macro avg       0.85      0.81      0.82     35817
weighted avg       0.85      0.84      0.84     35817
```
```python
precision_rf =  precision_score(y_test, y_pred_rf)
acc_rf = accuracy_score(y_test, y_pred_rf)
print('Precision: ',round(precision_rf * 100,4), '%')
print('Accuracy: ',round(acc_rf * 100,4), '%')
```
```
Precision:  87.2632 %
Accuracy:  84.4431 %
```

The random forest algorithm performs much better than KNN and logistic regression in terms of both precision and accuracy. The model will correctly predict the positives 87.26% of the time and an overall accuracy of 84.44%. 

![rf_impfeat](https://github.com/sangtvo/EDA-and-Hotel-Cancellation-Prediction/blob/main/images/rf_featimp.png?raw=true)
* Based on this top 10 important features, it is much different compared to the logistic regression model important features.
* The important features that should be targeted first are the types of deposit, lead time, special requests, market segment, and prior cancellations.

Solution
---
 MODELS | KNN | Logistic Regression | Random Forest
:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:
Precision | 77.17% | 86.02% | 87.43%
Accuracy | 81.44% | 79.97% | 84.50%
* The best model is the **random forest model** with an overall accuracy of 84.50% which means that the model will correctly predict hotel guest cancellation 84.50% of the time. 

This study has found the random forest model as the winning model and the recommended course of action for hotel management should be focusing at least 3 factors to reduce cancellation rates based on the top 10 important features. For any business, a good objective is for businesses to focus the larger and easier variables to mitigate their losses due to resources and manpower. 

It is evident that city cancellations are more dominant in the southern part of Portugal than resort hotels and average to high lead times have higher cancellation rates. . Perhaps city hotels should remodel or become attractive with amenities like a resort hotel, yet affordable for guests. As for lead times, the longer a guest holds a reservation, the more likely they are to cancel. In order to mitigate this risk, hotels can offer larger discounts or non-refundable bookings for booking in advance. Another option is to provide free amenities like breakfast, Wi-Fi, laundry or dry cleaning, and pool access. One factor that increases cancellation rates are deposit types. At least 80% of the data require no deposits. This is quite high and can be mitigated by requiring fees for cancellation or mandatory deposits. When hotels have stricter cancellation policies, guests are less inclined to cancel the reservation. 

Key Takeaways
---
* Roughly 75-80% of the guest reservations do not require deposits and a little over 33% of those guests cancel.
* 2016 is the most visited year which doubled compared to the previous year. 
* At least 50% of the bookings are made through an online travel agency and roughly 25% are offline travel agents and tour operators. 
* Aviation market segment don't bother booking resort hotels and city hotels are much more favorable.
* Roughly after 50 days, bookings are more likely to be cancelled. 
* Random forest model outperforms both KNN and logistic regression models with an overall accuracy of 84.50%. 
