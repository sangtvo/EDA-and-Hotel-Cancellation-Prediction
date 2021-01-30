# Capstone Project: EDA &amp; Hotel Cancellation Prediction
> This study analyzes hotel data from the southern part of Portugal to identify why guests are cancelling their reservations and the potential indicators that are causing them to do so. The original data is derived from Nuno Antonio, Ana Almeida, and Luis Nunes who are researchers from Lisbon University Institute and later cleaned/uploaded on [Kaggle](https://www.kaggle.com/jessemostipak/hotel-booking-demand). Python is the choice of program and will be using binary logistic regression to determine the most important indicators.

Table of Contents
---
1. [General Information](#general-information)
2. [Summary](#summary)
3. [Tech Stack](#tech-stack)
4. [Data Preprocessing/Cleaning](#data-preprocessingcleaning)
5. [Exploratory Data Analysis](#exploratory-data-analysis)
6. [Logistic Regression (Base)](#logistic-regression)
7. [Step-wise Logistic Regression](#step-wise-logistic-regression)
8. [Decision Tree](#decision-tree)
9. [Random Forest](#random-forest)
10. [Solution](#solution)
11. [Key Takeaways](#key-takeaways)

<a name="https://github.com/sangtvo/Customer-Churn-Analysis#general-information"/>
<a name="https://github.com/sangtvo/Customer-Churn-Analysis#summary"/>
<a name="https://github.com/sangtvo/Customer-Churn-Analysis#tech-stack"/>
<a name="https://github.com/sangtvo/Customer-Churn-Analysis#data-preprocessingcleaning"/>
<a name="https://github.com/sangtvo/Customer-Churn-Analysis#exploratory-data-analysis"/>
<a name="https://github.com/sangtvo/Customer-Churn-Analysis#logistic-regression"/>
<a name="https://github.com/sangtvo/Customer-Churn-Analysis#step-wise-logistic-regression"/>
<a name="https://github.com/sangtvo/Customer-Churn-Analysis#decision-tree"/>
<a name="https://github.com/sangtvo/Customer-Churn-Analysis#random-forest"/>
<a name="https://github.com/sangtvo/Customer-Churn-Analysis#solution"/>
<a name="https://github.com/sangtvo/Customer-Churn-Analysis#key-takeaways"/>

General Information
---
The capstone project is part of a graduate course in order to graduate at Western Governor's University. A completion of a capstone prospectus, executive summary, and a multimedia presentation is required to graduate, but will not be uploaded to my repository. 


Summary
---
The winning model is the **random forest** algorithm with an overall accuracy of 80.21% and AUC of 81.50%. This means that the model will correctly predict churn 80.21% of the time and there is a 81.50% chance that the model can distinguish between positive (churn) and negative (no churn) classes. In order for the telecommunications company to reduce the current churn rate of 26.58%, the company should focus on contracts (month-to-month specifically), tenure length, and total charges to start. The company should give incentives such as reduced pricing or discounted extra services for long-term customers to keep them from leaving to another competitor. 

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

Drop more variables after looking at correlation matrix.
```python
df=df.drop(columns=['reserved_room_type', 'assigned_room_type', 'agent'])
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


![Corr](https://github.com/sangtvo/EDA-and-Hotel-Cancellation-Prediction/blob/main/images/corr.png?raw=true)

Solution
---
 

Key Takeaways
---

