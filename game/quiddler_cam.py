from imutils.video import VideoStream, FPS
import numpy as np
from icevision.models import faster_rcnn
from icevision import tfms
from icevision.core import ClassMap
from icevision.data import Dataset
import cv2
from pathlib import Path
import torch

class_map = ClassMap(['a','b','c','d','e','f','g','h','i','j','k',
                'l','m','n','o','p','q','r','s','t','u','v',
                'w','x','y','z','qu','in','er','cl','th'])

def get_cards(pred, n=100):
    raw_scores = pred['scores']*100
    best_scores = raw_scores.argsort()[:-(n+1):-1]
    cards = np.array([class_map.get_id(id) for id in pred['labels']])[best_scores]
    scores = np.array([str(int(sc)) for sc in np.floor(raw_scores)])[best_scores]
    b_ord = np.array([bb.x for bb in pred['bboxes']])[best_scores].argsort()
    return ','.join(cards[b_ord]), ','.join(scores[b_ord]) 

size = 512

valid_tfms = tfms.A.Adapter([*tfms.A.resize_and_pad(size), tfms.A.Normalize()])

model = faster_rcnn.model(num_classes=len(class_map))

# Load a model
source = Path('/home/jeremy/Documents/models/quiddler')
model.load_state_dict(torch.load(source/'quiddler_model_24oct_512_1.pt'))
device = torch.device("cuda")
model.to(device)
model.eval()


vs = VideoStream(src=2).start()

fps = FPS().start()

while True:
    frame = vs.read()

    infer_ds = Dataset.from_images([frame], valid_tfms)

    batch, samples = faster_rcnn.build_infer_batch(infer_ds)
    preds = faster_rcnn.predict(model=model, batch=batch, detection_threshold=0.75)
    cards, scores = get_cards(preds[0],9)
    print(cards)
    print(scores)
    cv2.putText(frame, cards, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (50,255,50), 2)
    cv2.putText(frame, scores, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (50,255,50), 2)

    fps.update()
    cv2.imshow("Fast Cards", frame)
    key = cv2.waitKey(1) & 0xFF

    # Press q to quit
    if key == ord("q"):
        break

fps.stop()
print(f"FPS: {fps.fps():.2f}")

vs.stop()
cv2.destroyAllWindows()
