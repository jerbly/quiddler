import logging
import azure.functions as func
from icevision.all import *
import os
import json
import base64
import io

class_map = ClassMap(['a','b','c','d','e','f','g','h','i','j','k',
                'l','m','n','o','p','q','r','s','t','u','v',
                'w','x','y','z','qu','in','er','cl','th'])
size = 512

valid_tfms = tfms.A.Adapter([*tfms.A.resize_and_pad(size), tfms.A.Normalize()])
model = None

def get_cards(pred, n=100, img_b64=None):
    raw_scores = pred['scores']
    if len(raw_scores) == 0:
        return '','',0,0,''
    best_scores = raw_scores.argsort()[:-(n+1):-1]
    cards = np.array([class_map.get_id(id) for id in pred['labels']])[best_scores]
    scores = np.array([str(int(sc)) for sc in np.floor(raw_scores*100)])[best_scores]
    mean_score = np.mean(raw_scores[best_scores])
    min_score = np.min(raw_scores[best_scores])
    b_ord = np.array([bb.xyxy[0] for bb in pred['bboxes']])[best_scores].argsort()
    return '/'.join(cards[b_ord]), ','.join(scores[b_ord]), mean_score, min_score, img_b64

def predict_cards(img, n_cards):
    infer_ds = Dataset.from_images([img], valid_tfms)
    batch, samples = faster_rcnn.build_infer_batch(infer_ds)
    preds = faster_rcnn.predict(model=model, batch=batch, detection_threshold=0.5)

    f,p = plt.subplots(1,1)
    show_img(draw_pred(samples[0]["img"], preds[0], class_map=class_map, denormalize_fn=denormalize_imagenet), ax=p)
    img_bytes = io.BytesIO()
    f.savefig(img_bytes, format='png', bbox_inches="tight")
    img_b64 = base64.b64encode(img_bytes.getvalue()).decode("utf-8").replace("\n", "")
    img_b64 = f'data:image/png;base64,{img_b64}'

    return get_cards(preds[0], n_cards, img_b64)

def b64_to_np(d):
    JS_DATA_HEAD = 'data:image'
    if d.startswith(JS_DATA_HEAD):
        d = d[d.index(',')+1:]
    return np.frombuffer(base64.b64decode(d), np.uint8)

def raw_to_img(n):
    return cv2.cvtColor(cv2.imdecode(n, -1), cv2.COLOR_BGR2RGB)

def vec_to_img(r):
    return np.reshape(r, (480,640,3))

def main(req: func.HttpRequest, context: func.Context) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')
    global model
    if not model:
        model = faster_rcnn.model(num_classes=len(class_map))
        model_path = os.path.join(context.function_directory, 'quiddler.pt')
        model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))

    data = req.get_json()
    n_cards = data['n_cards']
    hand_vector = b64_to_np(data['hand'])
    deck_vector = b64_to_np(data['deck'])

    if data.get('images',False):
        hand = predict_cards(raw_to_img(hand_vector), n_cards)
        deck = predict_cards(raw_to_img(deck_vector), 1)
    else:
        hand = predict_cards(vec_to_img(hand_vector), n_cards)
        deck = predict_cards(vec_to_img(deck_vector), 1)
    return func.HttpResponse(json.dumps((hand[0], deck[0], hand[4], deck[4])))
    