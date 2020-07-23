# Machine-learning-Webapp-using-streamlit

Prerequisites : Python,Scikit-learn,Machine Learning

Streamlit is an open-source Python library that makes it easy to build beautiful custom web-apps for machine learning and data science.

# Code
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
