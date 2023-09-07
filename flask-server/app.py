from io import BytesIO
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing import image
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from joblib import load
import cv2
import tempfile
import urllib.request
from werkzeug.utils import secure_filename
import requests


import moviepy.editor as mp
import librosa
import tensorflow as tf

import boto3
from botocore.exceptions import NoCredentialsError

AWS_ACCESS_KEY = os.environ.get('AWS_SECRET_KEY_ID')
AWS_SECRET_KEY = os.environ.get('AWS_SECRET_KEY')
BUCKET_NAME = "can-detect-or-not-ah"  # Change this to your bucket name


def preprocess_video(video_path):
    video = mp.VideoFileClip(video_path)
    if video.audio is None:
        raise ValueError(f"No audio in video: {video_path}")
    audio = video.audio.to_soundarray()
    mfccs = librosa.feature.mfcc(y=audio[:, 0], sr=audio.shape[0], n_mfcc=13)
    avg_mfccs = np.mean(mfccs, axis=1)
    return avg_mfccs

# Importing deps for image prediction


def upload_to_aws(file, s3_file_name):
    s3 = boto3.client('s3', aws_access_key_id=AWS_ACCESS_KEY,
                      aws_secret_access_key=AWS_SECRET_KEY)
    try:
        s3.upload_fileobj(file, BUCKET_NAME, s3_file_name)
        print("Upload Successful")
        return True
    except NoCredentialsError:
        print("Credentials not available")
        return False


def delete_from_aws(s3_file):
    s3 = boto3.client('s3', aws_access_key_id=AWS_ACCESS_KEY,
                      aws_secret_access_key=AWS_SECRET_KEY)
    try:
        s3.delete_object(Bucket=BUCKET_NAME, Key=s3_file)
        print("Delete Successful")
        return True
    except Exception as e:
        print(f"An error occurred: {e}")
        return False


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# Set TensorFlow to run only on CPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
tf.config.experimental.set_visible_devices([], 'GPU')

image_model = load_model("./model/deepfake.h5")
audio_model = load('./model/audio.pkl')


app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})


@app.route("/")
def home():
    return {"message": "Test Backend Server"}


@app.route("/upload_video", methods=['POST'])
def upload_video():
    print(request.files)
    print(request.form)
    video_path = None

    if 'file' not in request.files:
        print('No file part.')
        # Handle the case for sample video
        if 'filepath' in request.form:
            video_path = request.form['filepath']
        else:
            return jsonify({"error": "No video part in the request."}), 400
    else:
        file = request.files['file']
        if file.filename == '':
            print('No video selected.')
            return jsonify({"error": "No video selected."}), 400

        try:
            file_obj = BytesIO(file.read())
            upload_to_aws(file_obj, file.filename)
            video_path = f"https://can-detect-or-not-ah.s3.ap-southeast-1.amazonaws.com/{file.filename}"
        except Exception as e:
            print(e)
            return jsonify({"error": str(e)}), 500

    if video_path:
        # Temporary file to download the video
        temp_video_file = tempfile.NamedTemporaryFile(delete=False).name
        try:
            urllib.request.urlretrieve(video_path, temp_video_file)
            cap = cv2.VideoCapture(temp_video_file)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frames_to_extract = np.random.choice(
                total_frames, min(50, total_frames), replace=False)  # Extract 250 random frames

            ai_count = 0
            human_count = 0
            image_score = 0

            for frame_no in frames_to_extract:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
                ret, frame = cap.read()
                if ret:
                    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(img)
                    img = img.resize((224, 224))
                    x = image.img_to_array(img)
                    x = np.expand_dims(x, axis=0)
                    x /= 255

                    prediction = image_model.predict(x)
                    image_score += prediction
                    if prediction < 0.5:
                        human_count += 1
                    else:
                        ai_count += 1
            image_score = image_score / len(frames_to_extract)
            # Make sure this also uses the temp video file
            features = preprocess_video(temp_video_file)
            prediction = audio_model.predict([features])
            print("Image Score: ", image_score)
            print("Audio Score: ", prediction)

            majority_vote = "AI Generated" if ai_count > human_count else "Human Generated"
            if majority_vote == "AI Generated" or prediction < 0.5:
                return jsonify({"message": "AI Generated"})
            else:
                return jsonify({"message": "Human Generated"})

        except Exception as e:
            print(e)
            return jsonify({"error": str(e)}), 500
        finally:
            # Clean up the temporary file
            if os.path.exists(temp_video_file):
                os.remove(temp_video_file)
    else:
        return jsonify({"error": "Unexpected error occurred."}), 500


@app.route("/upload_audio", methods=['POST'])
def upload_audio():
    print(request.files)
    print(request.form)
    audio_path = None

    if 'file' not in request.files:
        print('No file part.')
        if 'filepath' in request.form:  # Handle the case for sample audio
            audio_path = request.form['filepath']
        else:
            return jsonify({"error": "No audio part in the request."}), 400
    else:
        file = request.files['file']
        if file.filename == '':
            print('No audio selected.')
            return jsonify({"error": "No audio selected."}), 400

        try:
            file_obj = BytesIO(file.read())
            upload_to_aws(file_obj, file.filename)
            audio_path = f"https://can-detect-or-not-ah.s3.ap-southeast-1.amazonaws.com/{file.filename}"
        except Exception as e:
            print(e)
            return jsonify({"error": str(e)}), 500

    if audio_path:
        # Temporary file to download the audio
        temp_audio_file = tempfile.NamedTemporaryFile(delete=False).name
        try:
            urllib.request.urlretrieve(audio_path, temp_audio_file)
            # Assuming you'll need a function to preprocess audio similar to videos.
            # Placeholder. Needs actual implementation.
            features = preprocess_video(temp_audio_file)
            prediction = audio_model.predict([features])
            print(prediction)

            if prediction >= 0.5:
                return jsonify({"message": "Human Generated"})
            else:
                return jsonify({"message": "AI Generated"})

        except Exception as e:
            print(e)
            return jsonify({"error": str(e)}), 500
        finally:
            # Clean up the temporary file
            if os.path.exists(temp_audio_file):
                os.remove(temp_audio_file)
    else:
        return jsonify({"error": "Unexpected error occurred."}), 500


@app.route("/upload", methods=['POST'])
def upload():
    print(request.files)
    print(request.form)
    img_path = None

    if 'file' not in request.files:
        print('No file part.')
        # Handle the case for sample image
        if 'filepath' in request.form:
            img_path = request.form['filepath']
        else:
            return jsonify({"error": "No file part in the request."}), 400
    else:
        file = request.files['file']
        if file.filename == '':
            print('No file selected.')
            return jsonify({"error": "No file selected."}), 400

        try:
            file_obj = BytesIO(file.read())
            upload_to_aws(file_obj, file.filename)
            img_path = f"https://can-detect-or-not-ah.s3.ap-southeast-1.amazonaws.com/{file.filename}"
        except Exception as e:
            print(e)
            return jsonify({"error": str(e)}), 500

    if img_path:
        # Temporary file to download the image
        temp_img_file = tempfile.NamedTemporaryFile(delete=False).name
        try:
            urllib.request.urlretrieve(img_path, temp_img_file)
            img = Image.open(temp_img_file).resize((224, 224))
            x = np.asarray(img, dtype=np.float32)
            x = np.expand_dims(x, axis=0)
            x /= 255

            # Make the prediction
            prediction = image_model.predict(x)
            print(prediction)

            if prediction < 0.5:
                return jsonify({"message": "Human Generated"})
            else:
                return jsonify({"message": "AI Generated"})
        except Exception as e:
            print(e)
            return jsonify({"error": str(e)}), 500
        finally:
            # Clean up the temporary file
            if os.path.exists(temp_img_file):
                os.remove(temp_img_file)
    else:
        return jsonify({"error": "Unexpected error occurred."}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=False)
