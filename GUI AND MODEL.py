#!/usr/bin/env python
# coding: utf-8



# In[116]:


print(mapping_Location)
print(mapping_WindGustDir)
print(mapping_WindDir9am)
print(mapping_WindDir3pm)


# In[ ]:





# In[115]:


from tkinter import *
from tkinter import messagebox
import pandas as pd
# Tk class is used to create a root window
root = Tk()
root.geometry('800x500')
root.configure(background='lightblue')
root.title('Rain Predictor')
root.resizable(0, 0)

header = Label(root,text="Rain Predictor",bg='lightblue',
               fg='black',font=('Arial',18,'bold'))
header.pack()


frame1 = Label(root,bg='lightblue')
frame1.pack()

locationlb = Label(frame1,text="Location",bg='lightblue',
               fg='black',font=('Arial',12),padx=5)
locationlb.grid(row=0,column=0)
locationent=Entry(frame1,width=10,bg='white',
               fg='gray',font=('Arial',12),
                 borderwidth=1)
locationent.grid(row=0,column=1)

mintemplb = Label(frame1,text="Minimum Temperature",bg='lightblue',
               fg='black',font=('Arial',12),padx=5)
mintemplb.grid(row=0,column=2)
mintempent=Entry(frame1,width=10,bg='white',
               fg='gray',font=('Arial',12),
                 borderwidth=1)
mintempent.grid(row=0,column=3)

maxtemplb = Label(frame1,text="Maximum Temperature",bg='lightblue',
               fg='black',font=('Arial',12),padx=5)
maxtemplb.grid(row=0,column=4)
maxtempent=Entry(frame1,width=10,bg='white',
               fg='gray',font=('Arial',12),
                 borderwidth=1)
maxtempent.grid(row=0,column=5)



frame2 = Label(root,bg='lightblue')
frame2.pack()


rainfalllb = Label(frame2,text="Rainfall",bg='lightblue',
               fg='black',font=('Arial',12),padx=5)
rainfalllb.grid(row=0,column=0)
rainfallent=Entry(frame2,width=10,bg='white',
               fg='gray',font=('Arial',12),
                 borderwidth=1)
rainfallent.grid(row=0,column=1)

evaporationlb = Label(frame2,text="Evaporation",bg='lightblue',
               fg='black',font=('Arial',12),padx=5)
evaporationlb.grid(row=0,column=2)
evaporationent=Entry(frame2,width=10,bg='white',
               fg='gray',font=('Arial',12),
                 borderwidth=1)
evaporationent.grid(row=0,column=3)


sunshinelb = Label(frame2,text="Sunshine",bg='lightblue',
               fg='black',font=('Arial',12),padx=5)
sunshinelb.grid(row=0,column=4)
sunshineent=Entry(frame2,width=10,bg='white',
               fg='gray',font=('Arial',12),
                 borderwidth=1)
sunshineent.grid(row=0,column=5)


frame3 = Label(root,bg='lightblue')
frame3.pack()

windgustdirlb = Label(frame3,text="Wind Gust Direction",bg='lightblue',
               fg='black',font=('Arial',12),padx=5)
windgustdirlb.grid(row=0,column=0)
windgustdirent=Entry(frame3,width=10,bg='white',
               fg='gray',font=('Arial',12),
                 borderwidth=1)
windgustdirent.grid(row=0,column=1)

windgustspeedlb = Label(frame3,text="Wind Gust Speed",bg='lightblue',
               fg='black',font=('Arial',12),padx=5)
windgustspeedlb.grid(row=0,column=2)
windgustspeedent=Entry(frame3,width=10,bg='white',
               fg='gray',font=('Arial',12),
                 borderwidth=1)
windgustspeedent.grid(row=0,column=3)

winddir9lb = Label(frame3,text="Wind Direction At 9am",bg='lightblue',
               fg='black',font=('Arial',12),padx=5)
winddir9lb.grid(row=0,column=4)
winddir9ent=Entry(frame3,width=10,bg='white',
               fg='gray',font=('Arial',12),
                 borderwidth=1)
winddir9ent.grid(row=0,column=5)



frame4 = Label(root,bg='lightblue')
frame4.pack()

winddir3lb = Label(frame4,text="Wind Direction at 3pm",bg='lightblue',
               fg='black',font=('Arial',12),padx=5)
winddir3lb.grid(row=0,column=0)
winddir3ent=Entry(frame4,width=10,bg='white',
               fg='gray',font=('Arial',12),
                 borderwidth=1)
winddir3ent.grid(row=0,column=1)

windpeed9lb = Label(frame4,text="Wind Speed at 9am ",bg='lightblue',
               fg='black',font=('Arial',12),padx=5)
windpeed9lb.grid(row=0,column=2)
windspeed9ent=Entry(frame4,width=10,bg='white',
               fg='gray',font=('Arial',12),
                 borderwidth=1)
windspeed9ent.grid(row=0,column=3)

windspeed3lb = Label(frame4,text="Wind Speed at 3pm",bg='lightblue',
               fg='black',font=('Arial',12),padx=5)
windspeed3lb.grid(row=0,column=4)
windspeed3ent=Entry(frame4,width=10,bg='white',
               fg='gray',font=('Arial',12),
                 borderwidth=1)
windspeed3ent.grid(row=0,column=5)




frame5 = Label(root,bg='lightblue')
frame5.pack()

humidity9lb = Label(frame5,text="Humidity at 9am",bg='lightblue',
               fg='black',font=('Arial',12),padx=5)
humidity9lb.grid(row=0,column=0)
humidity9ent=Entry(frame5,width=10,bg='white',
               fg='gray',font=('Arial',12),
                 borderwidth=1)
humidity9ent.grid(row=0,column=1)

humidity3lb = Label(frame5,text="Humidity at 3pm",bg='lightblue',
               fg='black',font=('Arial',12),padx=5)
humidity3lb.grid(row=0,column=2)
humidity3ent=Entry(frame5,width=10,bg='white',
               fg='gray',font=('Arial',12),
                 borderwidth=1)
humidity3ent.grid(row=0,column=3)

pressure9lb = Label(frame5,text="Presuure at 9am",bg='lightblue',
               fg='black',font=('Arial',12),padx=5)
pressure9lb.grid(row=0,column=4)
pressure9ent=Entry(frame5,width=10,bg='white',
               fg='gray',font=('Arial',12),
                 borderwidth=1)
pressure9ent.grid(row=0,column=5)




frame6 = Label(root,bg='lightblue')
frame6.pack()

pressure3lb = Label(frame6,text="Pressure at 3pm",bg='lightblue',
               fg='black',font=('Arial',12),padx=5)
pressure3lb.grid(row=0,column=0)
pressure3ent=Entry(frame6,width=10,bg='white',
               fg='gray',font=('Arial',12),
                 borderwidth=1)
pressure3ent.grid(row=0,column=1)

cloud9lb = Label(frame6,text="Clouds at 9am",bg='lightblue',
               fg='black',font=('Arial',12),padx=5)
cloud9lb.grid(row=0,column=2)
cloud9ent=Entry(frame6,width=10,bg='white',
               fg='gray',font=('Arial',12),
                 borderwidth=1)
cloud9ent.grid(row=0,column=3)

cloud3lb = Label(frame6,text="Clouds at 3pm",bg='lightblue',
               fg='black',font=('Arial',12),padx=5)
cloud3lb.grid(row=0,column=4)
cloud3ent=Entry(frame6,width=10,bg='white',
               fg='gray',font=('Arial',12),
                 borderwidth=1)
cloud3ent.grid(row=0,column=5)



frame7 = Label(root,bg='lightblue')
frame7.pack()

temperature9lb = Label(frame7,text="Temperature at 9am",bg='lightblue',
               fg='black',font=('Arial',12),padx=5)
temperature9lb.grid(row=0,column=0)
temperature9ent=Entry(frame7,width=10,bg='white',
               fg='gray',font=('Arial',12),
                 borderwidth=1)
temperature9ent.grid(row=0,column=1)

temperature3lb = Label(frame7,text="Temperature at 3pm",bg='lightblue',
               fg='black',font=('Arial',12),padx=5)
temperature3lb.grid(row=0,column=2)
temperature3ent=Entry(frame7,width=10,bg='white',
               fg='gray',font=('Arial',12),
                 borderwidth=1)
temperature3ent.grid(row=0,column=3)

raintodaylb = Label(root,text="Did it rain today?",bg='lightblue',
               fg='black',font=('Arial',12),padx=5)
raintodaylb.pack()
options = StringVar()
options.set('No')
dropdown = OptionMenu(root, options,"Yes","No").pack()


predictionlb = Label(root,text="",bg='lightblue',
               fg='black',font=('Arial',12),padx=5)
predictionlb.pack(pady=20)




def encode_data(feature_name):
    mapping_dict = {}
    unique_values = list(rain[feature_name].unique())
    for idx in range(len(unique_values)):
        mapping_dict[unique_values[idx]] = idx
    return mapping_dict


def decode_data(mapping_dict, entry):
    return mapping_dict[entry]


def MLModel():
    import pandas as pd
    from sklearn import preprocessing
    from sklearn.linear_model import LogisticRegression 
    from sklearn.model_selection import train_test_split 
    from sklearn.preprocessing import StandardScaler

    df = pd.read_csv('cleaned_dataset.csv',index_col=0)
    df = pd.DataFrame(df)
    df.dropna(inplace=True)


    mapping_Location=encode_data('Location')
    mapping_WindGustDir=encode_data('WindGustDir')
    mapping_WindDir9am=encode_data('WindDir9am')
    mapping_WindDir3pm=encode_data('WindDir3pm')

    df['RainToday'].replace({'No':0, 'Yes': 1}, inplace = True)
    df['RainTomorrow'].replace({'No':0, 'Yes': 1}, inplace = True)
    df['WindGustDir'].replace(mapping_WindGustDir,inplace = True)
    df['WindDir9am'].replace(mapping_WindDir9am,inplace = True)
    df['WindDir3pm'].replace(mapping_WindDir3pm,inplace = True)
    df['Location'].replace(mapping_Location, inplace = True)


    X = df.drop(columns='RainTomorrow') 
    y = df.RainTomorrow 
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 0)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    classifier_logreg = LogisticRegression(solver='liblinear', random_state=0)
    classifier_logreg.fit(X_train, y_train)

    
    input_data=[]
    input_data.append(decode_data(mapping_Location,str(locationent.get())))
    input_data.append(float(mintempent.get()))
    input_data.append(float(maxtempent.get()))
    input_data.append(float(rainfallent.get()))
    input_data.append(float(evaporationent.get()))
    input_data.append(float(sunshineent.get()))
    input_data.append(decode_data(mapping_WindGustDir,str(windgustdirent.get())))
    input_data.append(float(windgustspeedent.get()))
    input_data.append(decode_data(mapping_WindDir9am,str(winddir9ent.get())))
    input_data.append(decode_data(mapping_WindDir3pm,str(winddir3ent.get())))
    input_data.append(float(windspeed9ent.get()))
    input_data.append(float(windspeed3ent.get()))
    input_data.append(float(humidity9ent.get()))
    input_data.append(float(humidity3ent.get()))
    input_data.append(float(pressure9ent.get()))
    input_data.append(float(pressure3ent.get()))
    input_data.append(float(cloud9ent.get()))
    input_data.append(float(cloud3ent.get()))
    input_data.append(float(temperature9ent.get()))
    input_data.append(float(temperature3ent.get()))
    if options.get()=='Yes':
        input_data.append(1)
    else:
        input_data.append(0)
    
    input_data = scaler.transform([input_data])
    predicted_value= classifier_logreg.predict(input_data)
    if (predicted_value == 1):
        predictionlb.config(text="It is clear tomorrow")
    else:
        predictionlb.config(text="High chances of rain tomorrow")
    

button = Button(text="Predict Rain",bg='white',
               activebackground='blue',borderwidth=1,
               font=('Arial',14),command=MLModel)
button.pack(pady=20)
root.mainloop()


# In[ ]:





# In[8]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
dataset = 'weatherAUS.csv'
rain = pd.read_csv(dataset)
rain['WindGustDir'].nunique()

