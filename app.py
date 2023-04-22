from flask import Flask, render_template, request
import numpy as np
import pickle

dia_type2 = pickle.load(open('models/dia-type2.pkl', 'rb'))
dia_type1 = pickle.load(open('models/dia-type1.pkl', 'rb'))
heart_model = pickle.load(open('models/heart.pkl', 'rb'))
liver_model = pickle.load(open('models/liver.pkl', 'rb'))


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route("/diabetes-report")
def diabetesReoprt():
    return render_template('report1.html')

@app.route("/heart-report")
def heartReport():
    return render_template('report2.html')

@app.route("/liver-report")
def liverReport():
    return render_template('report3.html')

@app.route("/diabetes", methods=['GET','POST'])
def diabetes():
    return render_template('diabetes.html')

@app.route("/type1", methods=['GET','POST'])
def type1():
    return render_template('type1.html')

@app.route("/heart", methods=['GET','POST'])
def heart():
    return render_template('heart.html')

@app.route("/liver", methods=['GET','POST'])
def liver():
    return render_template('liver.html')

@app.route("/predict", methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        if(len([float(x) for x in request.form.values()])==8):
            preg = int(request.form['pregnancies'])
            glucose = int(request.form['glucose'])
            bp = int(request.form['bloodpressure'])
            st = int(request.form['skinthickness'])
            insulin = int(request.form['insulin'])
            bmi = float(request.form['bmi'])
            dpf = float(request.form['dpf'])
            age = int(request.form['age'])
            
            data = np.array([[preg,glucose, bp, st, insulin, bmi, dpf, age]])
            my_prediction = dia_type2.predict(data)

            return render_template('predict.html', prediction=my_prediction, disease_name='type2 Diabetes')
        elif(len([float(x) for x in request.form.values()])==12):
            preg = int(request.form['pregnancies'])
            glucose = int(request.form['glucose'])
            bp = int(request.form['bloodpressure'])
            st = int(request.form['skinthickness'])
            insulin = int(request.form['insulin'])
            bmi = float(request.form['bmi'])
            dpf = float(request.form['dpf'])
            polyphagia = int(request.form['polyphagia'])
            visualblurring	 = int(request.form['visual blurring'])
            obesity = int(request.form['obesity'])
            smoker = int(request.form['smoker'])
            hdl = int(request.form['hdl'])

            data = np.array([[preg, glucose, bp, st, insulin, bmi, dpf, polyphagia, visualblurring, obesity, smoker, hdl]])
            my_type = dia_type1.predict(data)
            return render_template('predict.html', prediction=my_type, disease_name='Type1 Diabetes')

        elif(len([float(x) for x in request.form.values()])==10):
            Age = int(request.form['Age'])
            Total_Bilirubin = float(request.form['Total_Bilirubin'])
            Direct_Bilirubin = float(request.form['Direct_Bilirubin'])
            Alkaline_Phosphotase = int(request.form['Alkaline_Phosphotase'])
            Alamine_Aminotransferase = int(request.form['Alamine_Aminotransferase'])
            Aspartate_Aminotransferase = int(request.form['Aspartate_Aminotransferase'])
            Total_Protiens = float(request.form['Total_Protiens'])
            Albumin = float(request.form['Albumin'])
            Albumin_and_Globulin_Ratio = float(request.form['Albumin_and_Globulin_Ratio'])
            Gender_Male = int(request.form['Gender_Male'])

            data = np.array([[Age,Total_Bilirubin,Direct_Bilirubin,Alkaline_Phosphotase,Alamine_Aminotransferase,Aspartate_Aminotransferase,Total_Protiens,Albumin,Albumin_and_Globulin_Ratio,Gender_Male]])
            my_prediction = liver_model.predict(data)
            return render_template('predict.html', prediction=my_prediction, disease_name='Liver')

        elif(len([float(x) for x in request.form.values()])==13):
            age = int(request.form['age'])
            sex = int(request.form['sex'])
            cp = int(request.form['cp'])
            trestbps = int(request.form['trestbps'])
            chol = int(request.form['chol'])
            fbs = int(request.form['fbs'])
            restecg = int(request.form['restecg'])
            thalach = int(request.form['thalach'])
            exang = int(request.form['exang'])
            oldpeak = float(request.form['oldpeak'])
            slope = int(request.form['slope'])
            ca = int(request.form['ca'])
            thal = int(request.form['thal'])

            data = [age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal]
            data1 = np.array(data).reshape(1,-1)
            my_prediction = heart_model.predict(data1)
            return render_template('predict.html', prediction=my_prediction, disease_name='Heart')

if __name__ == "__main__":
    app.run(debug=True)