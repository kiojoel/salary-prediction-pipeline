from flask import Flask,request,render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import PredictPipeline, CustomData


app=Flask(__name__)

@app.route('/')
def index():
  return render_template('index.html')

@app.route('/predictdata',methods=['GET','POST'])
def predictdata():
  if request.method=='GET':
    return render_template('home.html')
  else:
    data = CustomData(
      age=float(request.form.get('age')),
      gender=request.form.get('gender'),
      education=request.form.get('education'),
      job=request.form.get('job'),
      experience=float(request.form.get('experience'))
    )
    pred_df = data.get_data_as_dataframe()
    print(pred_df)
    predict_pipeline = PredictPipeline()
    results = predict_pipeline.predict(pred_df)
    return render_template('home.html',results=results[0])


if __name__ == "__main__":
  app.run(debug=True)