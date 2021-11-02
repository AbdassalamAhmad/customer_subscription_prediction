import requests
import numpy as np


url = 'http://localhost:9696/predict'


customer = {
    "age" : 68,
    "job" : "retired",
    "martial" : "divorced",
    "education" : "secondary",
    "default" : "no",
    "balance_logs" : np.log1p(3000),
    "housing" : "no",
    "loan" : "no",
    "contact" : "telephone",
    "day" : 14,
    "month" : "jul",
    "duration" : 897,
    "campaign" : 2,
    "pdays" : -1,
    "previous" : 0,
    "poutcome" : "unknown"    
}


result = requests.post(url, json = customer).json()

print (result)



