# This is a sample Python script.
import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error,r2_score
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.
#Load Data & Data Transformation
data=pd.read_csv("D:\worldwide.csv")
print(data)
#Exploring Data
head=data.head()
tail=data.tail()
nulls=data.isnull().sum()
print(head)
print(tail)
print(nulls)
#Data Droping
list_of_colname=["Country/Region"]
droped_row=data.loc[4,list_of_colname]
print(droped_row)
# data.drop(droped_row)
# print(data)
#Now We find That We Have No Null Cell To Remove
#We Will Start Get Corr Matrix To Detarmine Which Col(Variable/Attrbuide) We want to simplify calc
list=["Rank","Cost index","Purchasing power index"]
print(data.loc[:,list].corr())
sns.heatmap(data.loc[:,list].corr())
#Start(Qantntive data) Visualization Data (Uni var)
sns.barplot(data["Rank"])
plt.show()
sns.barplot(data["Cost index"])
plt.show()
sns.barplot(data["Purchasing power index"])
plt.show()
#Start (Qantntive data) Visualization Data (Bivar)
plt.plot(data["Cost index"],data["Rank"])
plt.xlabel("Cost")
plt.ylabel("Rank")
plt.show()
#We Conclde That We Can Use Linear Regrssion Model
#Start Using Linear Regression Model
#Targer X And Y
list=["Cost index","Purchasing power index"]
x=data.loc[:,list]
X=np.array(x)
y=np.array(data["Rank"])
#Split Our Data
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.33,random_state=42)
#Start Train Model
lin_model=LinearRegression()
lin_model.fit(X_train,y_train)
#Print Thetas Of Eqn
print(lin_model.coef_)
# Now let Us test Our Model
Y_pred=lin_model.predict(X_test)
print(Y_pred)
#Discuss Error
error=mean_squared_error(Y_pred,y_test)
print(np.sqrt(error))
#Finally We Have Accepted Error And Our Model Sucsess to Predict





# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
