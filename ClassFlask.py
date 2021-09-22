
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
import sys
import cv2 as cv
import numpy as np
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import time
from playsound import playsound
from flask import Flask, Response, render_template
from flask.views import MethodView
import keyboard
from test_routes import cultureConfig, cultureDataset


# Root directory of the project
ROOT_DIR = os.path.abspath("../../")
# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import visualize
from mrcnn.model import MaskRCNN as modellib

app = Flask(__name__)


class InferenceConfig(cultureConfig().__class__):
    # Run detection on one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


class PetriCounter(MethodView):

    def get(self):
        self.weights_path = "ModelWeights/mask_rcnn_culture_0010.h5"
        self.root_dir = os.path.abspath("../")

        database_time_start = time.perf_counter()
        dataset_dir = "Dataset"
        # Load validation dataset
        self.dataset = cultureDataset()
        self.dataset.load_culture(dataset_dir, "val")
        # Create model in inference mode
        self.device = "/cpu:0"  # /cpu:0 or /gpu:0

        # Must call before using the dataset
        self.dataset.prepare()
        database_time_end = time.perf_counter()
        print(f'Model prep took {database_time_end - database_time_start} seconds using the {self.device}')

        weights_loading_time_start = time.perf_counter()
        with tf.device(self.device):
            self.model = modellib(mode="inference", model_dir=self.weights_path, config=InferenceConfig())


        self.model.load_weights(self.weights_path, by_name=True)
        weights_loading_time_end = time.perf_counter()

        print(f'Weights loading took {weights_loading_time_end - weights_loading_time_start} seconds using the {self.device}')


    def video_capture(self):
        # Video capture
        video_time = time.perf_counter()
        vcapture = cv.VideoCapture(0, cv.CAP_DSHOW)
        video_end = time.perf_counter()

        print(f'Video Loading took {video_end - video_time} seconds on the {self.device}')

        while True:
            success, image = vcapture.read()

            # If ESC pressed, stop the loop
            if keyboard.is_pressed('esc'):
                # ESC pressed
                print("Escape hit, closing...")
                break

            # If Space pressed, take a photo and process.
            if keyboard.is_pressed('space'):
                # OpenCV returns images as BGR, convert to RGB
                image = image[..., ::-1]
                print(image.shape)
                # Detect objects
                r = self.model.detect([image], verbose=0)[0]
                print("Start Visualisation")
                masked_image = visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], self.dataset.class_names, r['scores'])
                masked_image = cv.putText(img=np.float32(masked_image), text=f'{r["masks"].shape[2]} colonies', org=(50, 50),
                                          fontFace=cv.FONT_HERSHEY_COMPLEX, fontScale=3, lineType=cv.LINE_AA, color=(256, 0, 0))

                masked_image = cv.cvtColor(masked_image, cv.COLOR_RGB2BGR)

                show_analysis = True
                while show_analysis:
                    ret, jpeg = cv.imencode('.jpg', masked_image)
                    frame = jpeg.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

                    if keyboard.is_pressed("a"):
                        show_analysis = False

            else:
                ret, jpeg = cv.imencode('.jpg', image)
                frame = jpeg.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

        return render_template("Camera.html", camera=Response(self.video_capture(), mimetype='multipart/x-mixed-replace; boundary=frame'))


app.add_url_rule('/', view_func=PetriCounter.as_view("Colony"))