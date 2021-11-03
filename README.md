# customer_subscribtion_prediction Project
This is the mid-term project of the [Machine Learning Zoomcamp](https://github.com/alexeygrigorev/mlbookcamp-code/tree/master/course-zoomcamp) held by [DataTalksClub](https://datatalks.club/).
I did a complete ML engineer job from analyzing , preparing the data then training and tuning multiple ML models and finally deploying the best model(XGBoost) with docker locally and pythonanywhere on the cloud. If you want to check my other small projects see [this Repo](https://github.com/AbdassalamAhmad/Machine-Learning-Zoomcamp)
## Problem Description
The problem in this project is the prediction of whether potential bank clients will subscribe to the term deposit or not ('yes' or 'no' condition).
## About the Dataset
### Data Source
I use the dataset from here: https://archive.ics.uci.edu/ml/datasets/Bank+Marketing, or you can download the dataset for training and validating from this project repo [here](https://raw.githubusercontent.com/AbdassalamAhmad/customer_subscribtion_prediction/main/bank-full.csv). and the testing data [here](https://raw.githubusercontent.com/AbdassalamAhmad/customer_subscribtion_prediction/main/bank.csv)
### Dataset Information
The data is related with direct marketing campaigns of a Portuguese banking institution. The marketing campaigns were based on phone calls. Often, more than one contact to the same client was required, in order to access if the product (bank term deposit) would be ('yes') or not ('no') subscribed.

There are four datasets:(I've used the last two datasets, bank-full.csv for training and validating and bank.csv for final testing.
1. bank-additional-full.csv with all examples (41188) and 20 inputs, ordered by date (from May 2008 to November 2010), very close to the data analyzed in [Moro et al., 2014]
2. bank-additional.csv with 10% of the examples (4119), randomly selected from 1), and 20 inputs.
3. bank-full.csv with all examples and 17 inputs, ordered by date (older version of this dataset with less inputs).
4. bank.csv with 10% of the examples and 17 inputs, randomly selected from 3 (older version of this dataset with less inputs).
The smallest datasets are provided to test more computationally demanding machine learning algorithms (e.g., SVM).<br>
### Features Information
1. age (numeric)
2. job : type of job (categorical: "admin.","unknown","unemployed","management","housemaid","entrepreneur","student",      "blue-collar","self-employed","retired","technician","services")
3. marital : marital status (categorical: "married","divorced","single"; note: "divorced" means divorced or widowed)
4. education (categorical: "unknown","secondary","primary","tertiary")
5. default: has credit in default? (binary: "yes","no")
6. balance: average yearly balance, in euros (numeric) 
7. housing: has housing loan? (binary: "yes","no")
8. loan: has personal loan? (binary: "yes","no")
  related with the last contact of the current campaign:
9. contact: contact communication type (categorical: "unknown","telephone","cellular")
10. day: last contact day of the month (numeric)
11. month: last contact month of year (categorical: "jan", "feb", "mar", ..., "nov", "dec")
12. duration: last contact duration, in seconds (numeric)<br>
**other attributes:**
13. campaign: number of contacts performed during this campaign and for this client (numeric, includes last contact)
14. pdays: number of days that passed by after the client was last contacted from a previous campaign (numeric, -1 means client was not previously contacted)
15. previous: number of contacts performed before this campaign and for this client (numeric)
16. poutcome: outcome of the previous marketing campaign (categorical: "unknown","other","failure","success")<br>
**output variable (desired target):**
17. y - has the client subscribed a term deposit? (binary: "yes","no")

**Citation Request:**
This dataset is public available for research. The details are described in [Moro et al., 2014].
Please include this citation if you plan to use this database:
[Moro et al., 2014] S. Moro, P. Cortez and P. Rita. A Data-Driven Approach to Predict the Success of Bank Telemarketing. Decision Support Systems, Elsevier, 62:22-31, June 2014
###








