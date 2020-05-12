#!/usr/bin/env python
# coding: utf-8




import numpy as np
from flask import Flask, request,jsonify,render_template
import pickle





app=Flask(__name__)
model=pickle.load(open('Banking.pkl','rb'))



@app.route('/')
def home():   
    return render_template("index.html")
@app.route('/predict',methods=["GET","Post"])

def predict():
    final=[]
    int_features  = [(x) for x in request.form.values()]
    print(int_features)
    final.append(int_features[0])
    
    if(int_features[1]=="Admin"):
        l=0
        final.append(l)
    elif(int_features[1]=="Blue collar"):
        l=1
        final.append(l)
    elif(int_features[1]=="Entrepreneur"):
        l=2
        final.append(l)
    elif(int_features[1]=="Housemaid"):
        l=3 
        final.append(l)
    elif(int_features[1]=="Management"):
        l=4
        final.append(l)
    elif(int_features[1]=="Retired"):
        l=5
        final.append(l)
    elif(int_features[1]=="services"):
        l=7
        final.append(l)
    elif(int_features[1]=="Student"):
        l=8
        final.append(l)
    elif(int_features[1]=="Technican"):
        l=9
        final.append(l)
    elif(int_features[1]=="Unemployed"):
        l=10
        final.append(l)
    elif(int_features[1]=="Self employement"):
        l=6
        final.append(l)
    elif(int_features[1]=="None"):
        l=11
        final.append(l)
    else:
        l=11
        final.append(l)
    if(int_features[2]=="None"):
        k=0
        final.append(k)
    elif(int_features[2]=="Divorced"):
        k=3
        final.append(k)
    elif(int_features[2]=="Married"):
        k=1
        final.append(k)
    elif(int_features[2]=="Single"):
        k=2
        final.append(k)
    else:
        k=0
        final.append(k)
   
    if(int_features[3]=="Basic"):
        d=1
        final.append(d)
    elif(int_features[3]=="High school"):
        d=2
        final.append(d)
    elif(int_features[3]=="illiterate"):
        d=3
        final.append(d)
    elif(int_features[3]=="Professional course"):
        d=4
        final.append(d)
    elif(int_features[3]=="university degree"):
        d=5
        final.append(d)
    elif(int_features[3]=="None"):
        d=11
        final.append(d)
    else:
        d=11
        final.append(d)
    if(int_features[4]=="No"):
        o=1
        final.append(o)
    elif(int_features[4]=="yes"):
        o=2
        final.append(o)
    elif(int_features[4]=="None"):
        o=0
        final.append(o)
    else:
        o=0
        final.append(o)
    if(int_features[5]=="yes"):
        f=2
        final.append(f)
    elif(int_features[5]=="No"):
        f=1
        final.append(f)
    elif(int_features[5]=="None"):
        f=0
        final.append(f)
    else:
        f=0
        final.append(f)
    if(int_features[6]=="yes"):
        m=2
        final.append(m)
    elif(int_features[6]=="No"):
        m=1
        final.append(m)
    elif(int_features[6]=="None"):
        m=0
        final.append(m)
    else:
        m=0
        final.append(m)
    if(int_features[7]=="Telephone"):
        n=1
        final.append(n)
    elif(int_features[7]=="Cellular"):
        n=0
        final.append(n)
    else:
        n=1
        final.append(n)
    if(int_features[8]=="jan"):
        w=11
        final.append(w)
    elif(int_features[8]=="Feb"):
        w=10
        final.append(w)
    elif(int_features[8]=="Mar"):
        w=5
        final.append(w)
    elif(int_features[8]=="Apr"):
        w=0
        final.append(w)
    elif(int_features[8]=="May"):
        w=6
        final.append(w)
    elif(int_features[8]=="June"):
        w=4
        final.append(w)
    elif(int_features[8]=="July"):
        w=3
        final.append(w)
    elif(int_features[8]=="Aug"):
        w=1
        final.append(w)
    elif(int_features[8]=="Sep"):
        w=9
        final.append(w)
    elif(int_features[8]=="oct"):
        w=8
        final.append(w)
    elif(int_features[8]=="Nov"):
        w=7
        final.append(w)
    elif(int_features[8]=="Dec"):
        w=2
        final.append(w)
    else:
        w=6
        final.append(w)
    if(int_features[9]=="Mon"):
        b=1
        final.append(b)
    elif(int_features[9]=="Tue"):
        b=3
        final.append(b)
    elif(int_features[9]=="Wed"):
        b=4
        final.append(b)
    elif(int_features[9]=="Thu"):
        b=2
        final.append(b)
    elif(int_features[9]=="Fri"):
        b=0
        final.append(b)
    else:
        b=0
        final.append(b)
    print(final)
    final.append(int_features[10])
    final.append(int_features[11])
    final.append(int_features[12])
    final.append(int_features[13])
    if(int_features[14]=="Failure"):
        z=0
    elif(int_features[14]=="success"):
        z=2
    elif(int_features[14]=="Nonexistent"):
        z=1
    else:
        z=1
    final.append(z)
    final.append(int_features[15])
    final.append(int_features[16])
    final.append(int_features[17])
    final.append(int_features[18])
    final.append(int_features[19])
    end=[np.array(final)]
    prediction = model.predict(end)
    if(prediction==0):
        return render_template('index.html',prediction_text="Sorry no chance for the subsription of bank")
    elif(prediction==1):
        return render_template('index.html',prediction_text="Yes chance for the subsription of bank")
        



if __name__ == '__main__':
   app.run(host='0.0.0.0', port=8080)

