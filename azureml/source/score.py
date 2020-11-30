# from azureml.contrib.services.aml_request import AMLRequest, rawhttp
# from azureml.contrib.services.aml_response import AMLResponse
from icevision.all import *
import os
import json
import base64
from quiddler_game import Quiddler
# from inference_schema.schema_decorators import input_schema, output_schema
# from inference_schema.parameter_types.standard_py_parameter_type import StandardPythonParameterType

class_map = ClassMap(['a','b','c','d','e','f','g','h','i','j','k',
                'l','m','n','o','p','q','r','s','t','u','v',
                'w','x','y','z','qu','in','er','cl','th'])
size = 512

valid_tfms = tfms.A.Adapter([*tfms.A.resize_and_pad(size), tfms.A.Normalize()])

model = faster_rcnn.model(num_classes=len(class_map))

quiddler = Quiddler(vocab_path='/var/azureml-app/source/sowpods.txt')

def get_cards(pred, n=100):
    raw_scores = pred['scores']
    best_scores = raw_scores.argsort()[:-(n+1):-1]
    cards = np.array([class_map.get_id(id) for id in pred['labels']])[best_scores]
    scores = np.array([str(int(sc)) for sc in np.floor(raw_scores*100)])[best_scores]
    mean_score = np.mean(raw_scores[best_scores])
    min_score = np.min(raw_scores[best_scores])
    b_ord = np.array([bb.x for bb in pred['bboxes']])[best_scores].argsort()
    return '/'.join(cards[b_ord]), ','.join(scores[b_ord]), mean_score, min_score

def predict_cards(img, n_cards):
    infer_ds = Dataset.from_images([img], valid_tfms)
    batch, samples = faster_rcnn.build_infer_batch(infer_ds)
    preds = faster_rcnn.predict(model=model, batch=batch, detection_threshold=0.5)
    return get_cards(preds[0], n_cards)
    
def init():
    print("This is init()")
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'quiddler.pt')
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))

def run(data):
    try:
        data = json.loads(data)
        n_cards = data['n_cards']
        img = cv2.cvtColor(cv2.imdecode(np.frombuffer(base64.b64decode(data['hand']), np.uint8), -1), cv2.COLOR_BGR2RGB)
        hand = predict_cards(img, n_cards)
        img = cv2.cvtColor(cv2.imdecode(np.frombuffer(base64.b64decode(data['deck']), np.uint8), -1), cv2.COLOR_BGR2RGB)
        deck = predict_cards(img, 1)
        return hand[0], deck[0], quiddler.get_best_play(hand[0].split('/'),deck[0])

    except Exception as e:
        error = str(e)
        return error