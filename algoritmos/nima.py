from subprocess import Popen, PIPE, STDOUT
from os import  path
from iqa.src.utils import utils
import tensorflow as tf
import tensorflow.keras as keras
import grpc
from tensorflow_serving.apis import predict_pb2, prediction_service_pb2_grpc
import numpy as np
import json


TFS_HOST = 'localhost'
TFS_PORT = 8500

nimaPath = path.abspath("../image-quality-assessment/")
aestheticWeigth = "weights_mobilenet_aesthetic_0.07"
technicalWeigth = "weights_mobilenet_technical_0.11"
tf.get_logger().setLevel('ERROR')


def classify(aesthetic,path):
    if aesthetic:
        nimaExec = "sudo %s/predict  --docker-image nima-cpu --base-model-name MobileNet --weights-file %s/models/MobileNet/%s.hdf5 --image-source %s" % (nimaPath,nimaPath,aestheticWeigth,path)
    else:
        nimaExec = "sudo %s/predict  --docker-image nima-cpu --base-model-name MobileNet --weights-file %s/models/MobileNet/%s.hdf5 --image-source %s" % (nimaPath,nimaPath,technicalWeigth,path)
    p = Popen(nimaExec.split(), stdin=PIPE, stdout=PIPE, stderr=STDOUT)
    output = p.stdout.read()
    return output

def calc_mean_score(score_dist):
    score_dist = normalize_labels(score_dist)
    return (score_dist * np.arange(1, 11)).sum()
    
def normalize_labels(labels):
    labels_np = np.array(labels)
    return labels_np / labels_np.sum()

def preProcessImage(image):
    # Load and preprocess image with multi processing
    img = utils.load_image(image, target_size=(224, 224))
    return keras.applications.mobilenet.preprocess_input(img)

def _loadAndPreprocessImage(image):
    # Load and preprocess image with multi processing
    img = utils.load_image(image["path"], target_size=(224, 224))
    image["img"] = keras.applications.mobilenet.preprocess_input(img)
    return image


def loadAndPreprocessImage(images):
    # Load and preprocess image
    print("Load and Pre Process Nima Images")
    i = 0
    j = len(images)
    for image in images:
        print("(" + str(i+1) + " / " + str(j) + ")", end="\r")
        img = utils.load_image(image["path"], target_size=(224, 224))
        image["img"] = keras.applications.mobilenet.preprocess_input(img)
        i = i + 1
    print("\n")
    return image

def getImageQuality(image, modelName):
    # Run through model
    target = f'{TFS_HOST}:{TFS_PORT}'
    channel = grpc.insecure_channel(target)
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
    request = predict_pb2.PredictRequest()
    request.model_spec.name = modelName
    request.model_spec.signature_name = 'image_quality'

    request.inputs['input_image'].CopyFrom(
        tf.make_tensor_proto(np.expand_dims(image, 0))
    )

    response = stub.Predict(request, 10.0)
    result = round(calc_mean_score(response.outputs['quality_prediction'].float_val), 2)
    # print(json.dumps({'mean_score_prediction': np.round(result, 3)}, indent=2))
    return result