import os
from flask import Flask, request, render_template, send_file, send_from_directory
from werkzeug.utils import secure_filename
from models import solov2, condinst, queryinst, mask_rcnn, yolov8, hover_net

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads/'
TMP_FOLDER = 'models/tmp/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['TMP_FOLDER'] = TMP_FOLDER

models = {
    'solov2': solov2,
    'yolov8': yolov8,
    'condinst': condinst,
    'queryinst': queryinst,
    'mask_rcnn': mask_rcnn,
    'hovernet' : hover_net
}

@app.route('/models/tmp/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['TMP_FOLDER'], filename)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return 'No file part', 400
    file = request.files['file']
    if file.filename == '':
        return 'No selected file', 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    model_name = request.form.get('model')
    model = models.get(model_name)

    if not model:
        return 'Invalid model selected', 400


    result_filepath = model.predict(filepath)

    return send_file(result_filepath, mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)
