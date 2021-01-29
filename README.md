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
Removed customerID variable since it is not necessary for the purpose of this analysis.
```r
cdf$customerID <- NULL
```

#### Recoding: 
Recode some of the categorical variables for simplicity.
```r
cdf$SeniorCitizen <- as.factor(mapvalues(cdf$SeniorCitizen, from=c("0","1"), to=c("No", "Yes")))
cdf$MultipleLines <- as.factor(mapvalues(cdf$MultipleLines, from=c("No phone service"), to=c("No")))

for (i in 9:14){
  cdf[,i] <- as.factor(mapvalues(cdf[,i], from=c("No internet service"), to=c("No")))
}
```

Recode the dependent variable as a factor in the clean data frame instead of characters.
```r
cdf[, 'Churn'] <- as.factor(cdf[, 'Churn'])
```

#### Missing Data:


**For the full notebook, please check out Customer Churn Analysis.Rmd or html file in the "code" folder.**

Exploratory Data Analysis
---
<table>
  <tr><th colspan=2>Univariate Analysis</th></tr>
  <tr><td><img src="https://github.com/sangtvo/EDA-and-Hotel-Cancellation-Prediction/blob/main/images/c_deposit_type.png?raw=true"> </td>
  <td><img src="https://github.com/sangtvo/EDA-and-Hotel-Cancellation-Prediction/blob/main/images/c_hoteltype.png?raw=true"> </td></tr>
  <tr><td> <img src="https://github.com/sangtvo/EDA-and-Hotel-Cancellation-Prediction/blob/main/images/c_repeatedguests.png?raw=true"> </td><td> <img src="https://github.com/sangtvo/EDA-and-Hotel-Cancellation-Prediction/blob/main/images/c_year.png?raw=true"> </td></tr>
  <tr><td> <img src="https://github.com/sangtvo/EDA-and-Hotel-Cancellation-Prediction/blob/main/images/hotel_type.png?raw=true"> </td><td> </td></tr>


</table>


Solution
---
 

Key Takeaways
---

