import time
from flask import Flask, Response, render_template
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
import cv2 as cv
import numpy as np
import keyboard
from playsound import playsound
from test_routes import create_model
from mrcnn import visualize


"""
To start the venv:
1. Move to d: 
2. cd to directory
3. "venv\Scripts\activate"
4. set FLASK_APP=HelloWorld.py
5. flask run
"""

# Instantiate the Flask class and assign it to the variable "app"
app = Flask(__name__)


# This is a called a "route"  and is used to redirect the user to a specific page or perform a particular
# function when the user visits a particular URL.
# Flask uses the method "route()" to bind functions to a URL. Whenever a user visits a URL, the method
# attached to it is executed.


video = cv.VideoCapture(0, cv.CAP_DSHOW)

from mrcnn.config import Config


class cultureConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "culture"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + culture

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 3

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9


def gen(video):
    model = create_model()
    success, image = video.read()
    while success:
        success, image = video.read()

        # If ESC pressed, stop the loop
        if keyboard.is_pressed('esc'):
            # ESC pressed
            print("Escape hit, closing...")
            break

        # If Space pressed, take a photo and process.
        if keyboard.is_pressed('space'):
            # OpenCV returns images as BGR, convert to RGB
            image = image[..., ::-1]
            # Detect objects
            r = model.detect([image], verbose=0)[0]

            masked_image = visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], ["BG", "culture"], r['scores'], show_bbox=True)
            masked_image = cv.putText(img=np.float32(masked_image), text=f'{r["masks"].shape[2]} colonies', org=(50, 50),
                                      fontFace=cv.FONT_HERSHEY_DUPLEX, fontScale=2, lineType=cv.LINE_AA, color=(256, 0, 0), thickness=2)

            masked_image = cv.cvtColor(masked_image, cv.COLOR_RGB2BGR)

            show_analysis = True
            while show_analysis:
                ret, jpeg = cv.imencode('.jpg', masked_image)
                frame = jpeg.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

                if keyboard.is_pressed("a"):
                    show_analysis = False

            # Save a copy of the slide that the user has decided to analyse.
            image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
            time_current = time.strftime("%Y%m%d-%H%M%S")
            cv.imwrite(f'ScannedImages\{time_current}.png', image)

        else:
            ret, jpeg = cv.imencode('.jpg', image)
            frame = jpeg.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


@app.route('/')
def index():
    return render_template("CameraTable.html", camera=Response(gen(video), mimetype='multipart/x-mixed-replace; boundary=frame'))


@app.route('/video_feed')
def video_feed():
    global video
    return Response(gen(video), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run()

