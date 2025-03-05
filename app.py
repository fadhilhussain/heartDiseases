from flask import Flask, render_template,request,url_for,send_file
from fpdf import FPDF
from src.pipeline.predict_pipeline import PredictPipeline,NewData


application = Flask(__name__)
app = application


@app.route('/',methods = ['GET','POST'])
def index():
    if request.method == 'POST':
        data = NewData(
            gender = int(request.form.get('gender')),
            age = int(request.form.get('age')),
            education = float(request.form.get('education')),
            currentSmoker = int(request.form.get('currentSmoker')),
            cigsPerDay = int(request.form.get('cigsPerDay')),
            prevalentHyp = int(request.form.get('prevalentHyp')),
            totChol = int(request.form.get('totChol')),
            sysBP = float(request.form.get('sysBP')),
            BMI = float(request.form.get('BMI')),
            heartRate = float(request.form.get('heartRate')),
            glucose = float(request.form.get('glucose'))
        )

        prediction_data = data.get_dataFrame()
        print(prediction_data)

        #iniilizing prediction pipeline 
        predict_pipline = PredictPipeline()
        result = predict_pipline.predict(prediction_data)

        pdf = FPDF()
        pdf.add_page()
        pdf.set_font('Arial',size=12)
        pdf.cell(200, 10, txt="CHD Prediction Report", ln=True, align='C')
        pdf.ln(10)
        pdf.cell(200, 10, txt=f"Gender: {data.gender}", ln=True)
        pdf.cell(200, 10, txt=f"Age: {data.age}", ln=True)
        pdf.cell(200, 10, txt=f"Education: {data.education}", ln=True)
        pdf.cell(200, 10, txt=f"Current Smoker: {data.currentSmoker}", ln=True)
        pdf.cell(200, 10, txt=f"Cigarettes Per Day: {data.cigsPerDay}", ln=True)
        pdf.cell(200, 10, txt=f"Prevalent Hypertension: {data.prevalentHyp}", ln=True)
        pdf.cell(200, 10, txt=f"Total Cholostrol: {data.totChol}", ln=True)
        pdf.cell(200, 10, txt=f"Systolic BP: {data.sysBP}", ln=True)
        pdf.cell(200, 10, txt=f"BMI: {data.BMI}", ln=True)
        pdf.cell(200, 10, txt=f"Heart Rate: {data.heartRate}", ln=True)
        pdf.cell(200, 10, txt=f"Glucose: {data.glucose}", ln=True)
        pdf.ln(10)
        if result[0] == 0:
            res = ' NO CHD !'
        elif result[0] == 1:
            res = ' YES CHD '
        pdf.set_text_color(255, 0, 0)
        pdf.cell(200, 10, txt=f"{res}", ln=True)
        pdf_path = "CHD_Prediction_Report.pdf"
        pdf.output(pdf_path)
        return send_file(pdf_path, as_attachment=True)


    return render_template('index.html')


if __name__ == '__main__':
    app.run(host="0.0.0.0",debug=True)



