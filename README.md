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
### short description of the files
1. [main_notebook.ipynb](https://github.com/AbdassalamAhmad/customer_subscribtion_prediction/blob/main/main_notebook.ipynb) - Data preparation, EDA , train multiple models, tuning their parameters and validating them on multiple metrics.
2. [Tuning_XGB_parameters.ipynb](https://github.com/AbdassalamAhmad/customer_subscribtion_prediction/blob/main/Tuning_XGB_parameters.ipynb) - Tuning XGBoost parameters manually to show every step after changing any parameter.
3. [Final_Test.ipynb](https://github.com/AbdassalamAhmad/customer_subscribtion_prediction/blob/main/Final_Test.ipynb) - picking the best model and training it on full dataset then testing it on the second dataset and then saving this model to [model_1.bin](https://github.com/AbdassalamAhmad/customer_subscribtion_prediction/blob/main/model_1.bin).
4. [train.py](https://github.com/AbdassalamAhmad/customer_subscribtion_prediction/blob/main/train.py) - is an enhanced and script version of [Final_Test.ipynb](https://github.com/AbdassalamAhmad/customer_subscribtion_prediction/blob/main/Final_Test.ipynb) where it do basiclly the same job (training the best model and save it using pickle)
5. [predict.py](https://github.com/AbdassalamAhmad/customer_subscribtion_prediction/blob/main/predict.py) -  to load the model from this file [model_1.bin](https://github.com/AbdassalamAhmad/customer_subscribtion_prediction/blob/main/model_1.bin) and serve it via a web service (Flask) and it waits for anyone on local host to send customer data to predict its desicion.
6. [predict-test.py](https://github.com/AbdassalamAhmad/customer_subscribtion_prediction/blob/main/predict-test.py) - it test a random customer to see its desicion by opening the web service we just created using the flask app.
7. [predict-test-pythonanywhere.py](https://github.com/AbdassalamAhmad/customer_subscribtion_prediction/blob/main/predict-test-pythonanywhere.py) - it do the same as previous one but on the cloud (pythonanywhere) where I uploaded the flask app to be ready for test anytime through this [link](http://abdassalam.pythonanywhere.com/predict). but you have to do so using (the .py) file not through browser.
8. [Pipfile](https://github.com/AbdassalamAhmad/customer_subscribtion_prediction/blob/main/Pipfile) and [Pipfile.lock](https://github.com/AbdassalamAhmad/customer_subscribtion_prediction/blob/main/Pipfile.lock) - Python package dependencies, in the pipfile you can find all necessary librares and packages to be able to run the scripts with no problem. I used pipenv and the details on how to use it is in the how to try this model section below.
9. [Dockerfile](https://github.com/AbdassalamAhmad/customer_subscribtion_prediction/blob/main/Dockerfile) - it is used to build Docker image locally.
## How to try this model
1. clone this repo to get all the code and dataset on your local machine.
2. install pipenv -which is a packaging tool that will help installing all dependencies- , use this command on your terminal.
```py
pip install pipenv
```
3. install all dependencies using pipenv by typing this command in your terminal **inside your cloned repo folder** 
```py
pipenv install
```
4. Testing: There are three different ways to test this project [locally or on the cloud (PythonAnywhere) or using Docker]
* Locally:
   1. open your terminal and run (predict.py) script , this will depoly a Flask webapp on your local machine with the best tuned model.
   2. open another terminal and run (predict-test.py) this will test a pre-written customer to see its desicion, it should tell you the probability of his desicion.
   3. (optional) if you want to test any other customer you should open bank.csv dataset, it contains all the test data and pick any customer and fill it in (predict-test.py) after opening it in you favorite IDE or in your notepad.
* On the Cloud (PythonAnywhere):
   1. open your terminal and run (predict-test-pythonanywhere.py) this will test a pre-written customer to see its desicion.
   2. (optional) you can also open the same file and change the customer features to get difeerent response.
* Using Docker:( you should have docker installed on your machine to follow this way)
   1. open your terminal in the directory of cloned repo on your local machine.
   2. build the Docker image locally using this command.
    ```py
    docker build -t zoomcamp-midterm_project .
    ```
   3. run the docker image using this command.
    ```py
    docker run -it --rm -p 9696:9696 zoomcamp-midterm_project
    ```
   4. open another terminal and run (predict-test.py) this will test a pre-written customer to see its desicion, it should tell you the probability of his desicion.
##






