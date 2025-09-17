from flask import Flask,request,render_template,jsonify
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import RobustScaler
application=Flask(__name__)
app=application
ensemble_model = pickle.load(open(r'D:\doanchuyennganh\Notebooks\Models\ensemble_model.pkl', 'rb'))
scaler = pickle.load(open(r'D:\doanchuyennganh\Notebooks\Models\scaler.pkl', 'rb'))
@app.route('/')
def home():
    return render_template('home.html')
@app.route('/index')
def index():
    return render_template('index.html')
@app.route('/predict_datapoint',methods=['GET','POST'])
def predict_datapoint():
    if request.method == 'POST':
        Pregnancies=float(request.form.get('Pregnancies'))
        Glucose=float(request.form.get('Glucose'))
        BloodPressure=float(request.form.get('BloodPressure'))
        SkinThickness=float(request.form.get('SkinThickness'))
        Insulin=float(request.form.get('Insulin'))
        BMI=float(request.form.get('BMI'))
        DiabetesPedigreeFunction=float(request.form.get('DiabetesPedigreeFunction'))
        Age=float(request.form.get('Age'))
        newdata_scaled=scaler.transform([[Pregnancies,	Glucose,	BloodPressure,	SkinThickness,	Insulin,	BMI	,DiabetesPedigreeFunction,	Age	]])
        input_dict = {
            'Pregnancies': [Pregnancies],
            'Glucose': [Glucose],
            'BloodPressure': [BloodPressure],
            'SkinThickness': [SkinThickness],
            'Insulin': [Insulin],
            'BMI': [BMI],
            'DiabetesPedigreeFunction': [DiabetesPedigreeFunction],
            'Age': [Age],
            'NewBMI_Obesity 1': [0],
            'NewBMI_Obesity 2': [0],
            'NewBMI_Obesity 3': [0],
            'NewBMI_Overweight': [0],
            'NewBMI_Underweight': [0],
            'NewInsulinScore_Normal': [0],
            'NewGlucose_Low': [0],
            'NewGlucose_Normal': [0],
            'NewGlucose_Overweight': [0],
            'NewGlucose_Secret': [0]
        }
        input_df = pd.DataFrame(input_dict)
        X_scaled = scaler.transform(input_df[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']])
        X_scaled = pd.DataFrame(X_scaled, columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])
        X_final = pd.concat([X_scaled, input_df.drop(['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'], axis=1)], axis=1)
        Predict=ensemble_model.predict(X_final)
        if Predict[0]==1:
            result='Bị tiểu đường'
        else:
            result='Không bị tiểu đường'

        return render_template('prediction.html',result=result)
    else:
        return render_template('home.html')
if __name__=="__main__":
    app.run(host="0.0.0.0",port=5501,debug=True)


