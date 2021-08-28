from flask import Flask, render_template, request, redirect
import pandas as pd 
from jcopml.utils import load_model

app = Flask(__name__)
model = load_model('model/diabetes.pkl')

@app.route('/', methods=['GET','POST'])
def index():
    if request.method == 'GET':
        return render_template('index.html')
    else:
        data = request.form.to_dict()
        X_test = pd.DataFrame([data])
        hasil = model.predict(X_test)
        return render_template('hasil.html', FinalData=hasil)

if __name__ == '__main__':
    app.run(host='127.0.0.1', port='5000', debug=True)
