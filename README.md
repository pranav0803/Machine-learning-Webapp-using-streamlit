# Machine-learning-Webapp-using-streamlit

Prerequisites : Python,Scikit-learn,Machine Learning

Streamlit is an open-source Python library that makes it easy to build beautiful custom web-apps for machine learning and data science.

Dataset: Mushrooms.csv

![Image](https://github.com/pranav0803/Machine-learning-Webapp-using-streamlit/blob/master/strem.PNG)

# Code

<h4>Import required libraries </h4>

```
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import precision_score, recall_score 

```

<h4> Function to Load Dataset and Encode values</h4>
As the data set is not in numeric format we use LabelEncoder which refers to converting the labels into numeric form so as to convert it into the machine-readable form

```
def load_data():
        data=pd.read_csv('mushrooms.csv')
        label =LabelEncoder()
        for col in data.columns:
            data[col]=label.fit_transform(data[col])
        return data
```

<h4> Function to Split the dataset into test and training </h4>

```
def split(df):
        y= df.type
        x= df.drop(columns=['type'])
        x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)
        return x_train,x_test,y_train,y_test
```


<h4>Function to Plotting and Evaluating metrics</h4>
We consider of using 1) Confusion Matrix 2)ROC Curve 3)Precision-Recall Curve

```
def plot_metrics(metrics_list):
        if 'Confusion Matrix' in metrics_list:
            st.subheader("Confusion Matrix")
            plot_confusion_matrix(model,x_test,y_test,display_labels=class_names)
            st.pyplot()
        if 'ROC Curve' in metrics_list:
            st.subheader("Roc Curve ")
            plot_roc_curve(model,x_test,y_test)
            st.pyplot()
        if 'Precision-Recall Curve' in metrics_list:
            st.subheader("Precision-Recall Curve")
            plot_precision_recall_curve(model,x_test,y_test)
            st.pyplot()
```

Run the webapp.py file in anaconda virtual environment after installing the required libraries
1) Streamlit --> pip install streamlit
2) sklearn  --> pip install sklearn

#### To Run
streamlit run webapp.py
