from flask import Flask, render_template,request,url_for,send_file
from fpdf import FPDF
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import PredictPipeline,NewData


application = Flask(__name__)
app = application

grid_model = pickle.load(open('models/grid1.pkl','rb'))
scaler = pickle.load(open('models/scaler1.pkl','rb'))


@app.route('/',methods = ['GET','POST'])
def index():
    if request.method == 'POST':
        gender = int(request.form.get('gender'))
        age = int(request.form.get('age'))
        education = float(request.form.get('education'))
        currentSmoker = int(request.form.get('currentSmoker'))
        cigsPerDay = int(request.form.get('cigsPerDay'))
        prevalentHyp = int(request.form.get('prevalentHyp'))
        totChol = int(request.form.get('totChol'))
        sysBP = float(request.form.get('sysBP'))
        BMI = float(request.form.get('BMI'))
        heartRate = float(request.form.get('heartRate'))
        glucose = float(request.form.get('glucose'))

        new_scaled = scaler.transform([[gender,age,currentSmoker,cigsPerDay,BPmeds,prevalentHyp,sysBP,diaBP,heartRate,glucose]])     
        # probs = grid_model.predict_proba(new_scaled)
        result = grid_model.predict(new_scaled)

        pdf = FPDF()
        pdf.add_page()
        pdf.set_font('Arial',size=12)
        pdf.cell(200, 10, txt="CHD Prediction Report", ln=True, align='C')
        pdf.ln(10)
        pdf.cell(200, 10, txt=f"Gender: {gender}", ln=True)
        pdf.cell(200, 10, txt=f"Age: {age}", ln=True)
        pdf.cell(200, 10, txt=f"Current Smoker: {currentSmoker}", ln=True)
        pdf.cell(200, 10, txt=f"Cigarettes Per Day: {cigsPerDay}", ln=True)
        pdf.cell(200, 10, txt=f"BP Medications: {BPmeds}", ln=True)
        pdf.cell(200, 10, txt=f"Prevalent Hypertension: {prevalentHyp}", ln=True)
        pdf.cell(200, 10, txt=f"Systolic BP: {sysBP}", ln=True)
        pdf.cell(200, 10, txt=f"Diastolic BP: {diaBP}", ln=True)
        pdf.cell(200, 10, txt=f"Heart Rate: {heartRate}", ln=True)
        pdf.cell(200, 10, txt=f"Glucose: {glucose}", ln=True)
        pdf.ln(10)
        if result[0] == 0:
            res = ' NO CHD !'
        elif result[0] == 1:
            res = ' YES CHD '
        pdf.set_text_color(255, 0, 0)
        pdf.cell(200, 10, txt=f"Prediction: {res}", ln=True)
        # pdf.cell(200, 10, txt=f"Probabilites towords to happen CHD: {probs}", ln=True)
        pdf_path = "CHD_Prediction_Report.pdf"
        pdf.output(pdf_path)
        return send_file(pdf_path, as_attachment=True)


    return render_template('index.html')


if __name__ == '__main__':
    app.run(host="0.0.0.0",debug=True)



