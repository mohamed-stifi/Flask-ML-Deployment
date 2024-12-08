from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from keras.applications.vgg16 import VGG16
from transformers import pipeline
from utils import *
import os
import pickle



app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.abspath('./data')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1000 * 1000
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

vgg16_model = VGG16(weights='imagenet')
print('vgg16_model Model loaded.')
reg_model = pickle.load(open('model.pkl','rb'))
print('reg_model Model loaded.')
gpt_pipeline = pipeline('text-generation', model="gpt2")
print('GPT-2 Pipeline loaded.')




@app.route('/predict', methods=['GET', 'POST'])
def redirect_example():
    if request.method == 'GET':
        return render_template('index.html')

    if request.method == 'POST':
        imagefile = request.files.get('imagefile', '')
        if imagefile == '':
            return render_template('index.html')

        else:
            img_name = secure_filename(imagefile.filename)
            img_path = os.path.join(app.config['UPLOAD_FOLDER'], img_name)
            imagefile.save(img_path)
            label = vgg16_predict(img_path, vgg16_model)

            classification = '%s (%.2f%%)' % (label[1], label[2]*100)
            return render_template('index.html', prediction=classification)



@app.route('/regpredict', methods=['GET', 'POST'])
def regpredict():
    if request.method == 'GET':
        return render_template('reg_index.html')

    if request.method == 'POST':
        try:
            experience = float(request.form['experience'])
            test_score = float(request.form['test_score'])
            interview_score = float(request.form['interview_score'])

            input_features = np.array([experience, test_score, interview_score]).reshape(1, -1)
            output = reg_model.predict(input_features)[0]
            return render_template('reg_index.html', prediction_text='Employee Salary should be $ {}'.format(output))
        except Exception as e:
            return render_template('reg_index.html', prediction_text=f'Error: {str(e)}')


@app.route('/textgen',  methods=['GET', 'POST'])
def textgen():
    if request.method == 'GET':
        return render_template('textgen_index.html')

    if request.method == 'POST':
        try:
            prompt = request.form['userInput']
            output = gpt_pipeline(prompt, do_sample=False)
            return render_template('textgen_index.html', generated_text = output[0]['generated_text'])
        except Exception as e:
            return render_template('textgen_index.html', generated_text = f'Error: {str(e)}')


@app.route('/home', methods=['GET'])
def home():
    return render_template('home.html')



if __name__ == '__main__': 
    app.run(host='0.0.0.0', port=5000, debug=False) 