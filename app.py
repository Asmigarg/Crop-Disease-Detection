from flask import Flask, request, jsonify, render_template
import os
from werkzeug.utils import secure_filename
from model import predict
app = Flask(__name__)

@app.route('/')
def home():
   return render_template('index.html')

@app.route('/take-photo')
def take_photo():
    return render_template('take-photo.html')


TEMP_DIR = 'temp'
os.makedirs(TEMP_DIR, exist_ok=True)

def clear_temp_folder(folder_path):
   """Delete all files in the given folder."""
   for filename in os.listdir(folder_path):
      file_path = os.path.join(folder_path, filename)
      try:
         if os.path.isfile(file_path):
            os.remove(file_path)
      except Exception as e:
         print(f"Error deleting file {file_path}: {e}")

@app.route('/upload', methods=['POST'])
def upload_image():
   if 'image' not in request.files:
      return jsonify({'error': 'No image file provided'}), 400

   image = request.files['image']
   if image.filename == '':
      return jsonify({'error': 'No selected file'}), 400

    # Clean up old images before saving the new one
   clear_temp_folder(TEMP_DIR)

   filename = secure_filename(image.filename)
   file_path = os.path.join(TEMP_DIR, filename)
   image.save(file_path)

   result = predict(file_path)

   return jsonify({'message': 'Image uploaded successfully', 'path': file_path, 'prediction': result})


if __name__ == '__main__':
   app.run()