from flask import Flask, render_template, request, redirect, flash, url_for
from ImageClassifier import ImageClassifier
from werkzeug.utils import secure_filename
import magic
import os
import cv2

UPLOAD_FOLDER = "/home/luis/Documents/src/static"
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

app = Flask(__name__)
app.secret_key = "Pa$$w0rd"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/', methods=['GET', 'POST'])
@app.route('/homepage', methods=['GET', 'POST'])
def upload_image():

    if request.method == "POST":

        if request.files:

            image = request.files["image"]

            if image.filename == "":

                # print("Selecione uma imagem!")
                return redirect(request.url)

            if not allowed_image(image.filename):

                # flash("A extensão do ficheiro selecionado não é suportada!")
                return redirect(request.url)
            else:

                filename = secure_filename(image.filename)
                image.save(os.path.join(app.config["UPLOAD_FOLDER"], filename))
                im = ImageClassifier()
                im.write_image(filename, im.classify(os.path.join(app.config["UPLOAD_FOLDER"], filename)), app.config["UPLOAD_FOLDER"])
                return render_template('result.html', filename = filename)  
                
            
    return render_template('homepage.html')

def allowed_image(filename):

    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
