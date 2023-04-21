import flask
from flask import Flask, render_template, request
import pickle
import numpy as np
import sklearn

app = Flask(__name__)

model = pickle.load(open('ssam.pkl','wb'))


@app.route('/')
def home():
    return render_template('rename.html')

@app.route('/getdata', methods=['POST'])

def pred():
    gender = request.form['Gender']
    print(gender)
    seniorcitizen = request.form['Senior_Citizen']
    print(seniorcitizen)
    dependents = request.form['Dependents']
    print(dependents)
    partner = request.form['Partner']
    print(partner)
    tenure = request.form['Tenure']
    print(tenure)
    phoneservice = request.form['Phone_Service']
    print(phoneservice)
    multiplelines = request.form['Multiple_Lines']
    print(multiplelines)
    onlinesecurity = request.form['Online_Security']
    print(onlinesecurity)
    onlinebackup = request.form['Online_Backup']
    print(onlinebackup)
    streamingtv = request.form['Streaming_TV']
    print(streamingtv)
    streamingmovies = request.form['Streaming_Movies']
    print(streamingmovies)
    paperlessbilling = request.form['Paper_less_Billing']
    print(paperlessbilling)
    churn = request.form['Churn']
    print(churn)
    contract = request.form['Contract']
    print(contract)
    internetservice = request.form['Internet_Service']
    print(internetservice)
    paymentmethod = request.form['Payment_Method']
    print(paymentmethod)
    deviceprotection = request.form['Device_Protection']
    print(deviceprotection)
    techsupport = request.form['Tech_Support']
    print(techsupport)
    inp_features = [[int(gender), int(seniorcitizen), int(partner), int(dependents), int(tenure), int(phoneservice),
                     int(multiplelines), int(internetservice),
                     int(onlinesecurity),
                     int(onlinebackup), int(deviceprotection), int(techsupport), int(streamingtv), int(streamingmovies), int(contract),
                     int(paperlessbilling), int(paymentmethod)]]
    print(inp_features)
    prediction = model.predict(inp_features)
    print(type(prediction))
    t = prediction[0]
    print(t)
    if t > 0.5:
        prediction_text = 'Customer will retain'
    else:
        prediction_text = 'Customer will not retain'
    print(prediction_text)
    return render_template('prediction.html', prediction_results=prediction_text)


if __name__ == "__main__":
    app.run()
