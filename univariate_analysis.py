# -*- coding: utf-8 -*-
"""Univariate Analysis.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1e2-vKSG3WDnqSCXolJU1AbqODxYHTGeX
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("cleaned_weatherAUS.csv")

df['RainTomorrow'].value_counts().plot(kind='bar',color={'pink','purple'})
df.dtypes

#Histograms
numerical_variables = df.select_dtypes(include=['number'])
numerical_variables = numerical_variables.drop(numerical_variables.columns[0], axis=1)
numerical_variables = numerical_variables.drop(numerical_variables.columns[-2], axis=1)
numerical_variables = numerical_variables.drop(numerical_variables.columns[-1], axis=1)

for var in numerical_variables:
    # Histogram
    plt.figure(figsize=(8, 4))
    plt.hist(df[var], bins=20, edgecolor='k')
    plt.xlabel(var)
    plt.ylabel('Frequency')
    plt.title(f'Histogram of {var}')
    plt.show()

#Box Plots
plt.figure(figsize=(15, 12))
for n, i in enumerate(numerical_variables):
    # add a new subplot iteratively
    ax = plt.subplot(6, 3, n + 1)
    plt.subplots_adjust(hspace=0.7)

    # filter df and plot on the new subplot axis
    sns.boxplot(numerical_variables[i],palette="rocket",orient="v",width=0.7,linewidth=4,ax=ax)

    # chart formatting
    ax.set_title(i.upper())
    ax.set_xlabel("")

#Pie Chart
categorical_variable = 'RainToday'

plt.figure(figsize=(6, 6))
df[categorical_variable].value_counts().plot(kind='pie', autopct='%1.1f%%')
plt.title(f'Pie Chart of {categorical_variable}')
plt.show()


categorical_variable = 'RainTomorrow'

plt.figure(figsize=(6, 6))
df[categorical_variable].value_counts().plot(kind='pie', autopct='%1.1f%%')
plt.title(f'Pie Chart of {categorical_variable}')
plt.show()

categorical_variable = 'WindDir3pm'

plt.figure(figsize=(6, 6))
df[categorical_variable].value_counts().plot(kind='pie', autopct='%1.1f%%')
plt.title(f'Pie Chart of {categorical_variable}')
plt.show()

#Density Plots
for var in numerical_variables:
    plt.figure(figsize=(8, 4))
    sns.kdeplot(numerical_variables[var],fill=True)
    plt.xlabel(var)
    plt.ylabel('Density')
    plt.title(f'Density Plot of {var}')
    plt.show()

#Non-numerical Univariate Plots
###count Location
plt.figure(figsize=(18,16))
ax = sns.countplot(x ='Location',data=df, palette='Set2')
plt.xticks(rotation = 90)
ax.bar_label(ax.containers[0])###Add a label to the height of the chart bars
plt.title('Location count', fontsize=18)

plt.show()