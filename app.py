import numpy as np
from flask import Flask, request, jsonify,url_for, redirect, render_template
import pickle
import pandas as pd
from werkzeug import secure_filename
import os
from preProcessFunc import create_model
from preProcessFunc import testFunc
from prepProcessFunctions import preprocessSteps
from prepProcessFunctions2 import identify_col_types_manual



os.chdir("/Users/nitinranjansharma/Documents/Nitin/Codes/Python/flaskDevApp/myProject/")
UPLOAD_FOLDER = '/data/'
app = Flask(__name__)
prediction = ""

@app.route('/', methods=['GET', 'POST'])
def col_define():
	if request.method == 'POST':
		colNames = [x for x in request.form.values()]
		print(colNames)
		identify_col_types_manual(colNames)
		render_template('indexSecond.html')
		return redirect(url_for('requirement'))
	return render_template('indexSecond.html')


@app.route('/requirement/', methods=['GET', 'POST'])
def requirement():
	if request.method == 'POST':
		features = [float(x) for x in request.form.values()]
		final_features = [(features)]
		prediction = testFunc(final_features)

    

		render_template('indexFirst.html', prediction_text='Success'.format(prediction))
		#return render_template('index.html')
		return redirect(url_for('upload'))
	return render_template('indexFirst.html')



@app.route('/requirement/upload/', methods=['GET', 'POST'])
def upload():
	if request.method == 'POST':
		df = request.files.get('file')
		
		#df = request.files.get('file')
		p = create_model(df)
		return render_template('index.html', dialog_output=p,prediction=p)
	return render_template('index.html')
	



if __name__ == "__main__":
	app.run(host='127.0.0.1', port=5000,debug=True)