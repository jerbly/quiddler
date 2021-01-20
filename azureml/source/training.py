import os
from icevision.all import *
from azureml.core.run import Run
import argparse
from utils import AzureRunLogCallback
import pandas as pd
import numpy as np

# Get the experiment run context
run = Run.get_context()

# Set hyperparameters
parser = argparse.ArgumentParser()
parser.add_argument('--input_data', type=str, dest="input_data")
parser.add_argument('--epochs', type=int, dest='epochs', default=30, help='epochs')
parser.add_argument('--batch_size', type=int, dest='batch_size', default=6, help='batch_size')
parser.add_argument('--image_size', type=int, dest='image_size', default=512, help='image_size')
args = parser.parse_args()
run.log('epochs', args.epochs)
run.log('batch_size', args.batch_size)
run.log('image_size', args.image_size)
run.log('input_data', args.input_data)

data_dir = args.input_data
print(f'Data dir = {data_dir}')

class_map = ClassMap(['a','b','c','d','e','f','g','h','i','j','k',
                'l','m','n','o','p','q','r','s','t','u','v',
                'w','x','y','z','qu','in','er','cl','th'])

source = Path(data_dir)/'train'
    
parser = parsers.via(source/'via.json', source, class_map)
train_rs, valid_rs = parser.parse(RandomSplitter([0.8, 0.2], seed=42))

size = args.image_size
presize = size+size//2

valid_tfms = tfms.A.Adapter([*tfms.A.resize_and_pad(size), tfms.A.Normalize()])
train_tfms = tfms.A.Adapter([*tfms.A.aug_tfms(size=size, presize=presize, horizontal_flip=None), tfms.A.Normalize()])

train_ds = Dataset(train_rs, train_tfms)
valid_ds = Dataset(valid_rs, valid_tfms)

model = faster_rcnn.model(num_classes=len(class_map))

# Train the model
print('Training the model...')

train_dl = faster_rcnn.train_dl(train_ds, batch_size=args.batch_size, num_workers=0, shuffle=True)
valid_dl = faster_rcnn.valid_dl(valid_ds, batch_size=args.batch_size, num_workers=0, shuffle=False)

metrics = [COCOMetric(metric_type=COCOMetricType.bbox)]

learn = faster_rcnn.fastai.learner(dls=[train_dl, valid_dl], model=model, metrics=metrics, 
                                   cbs=[AzureRunLogCallback(run)])

learn.fine_tune(args.epochs, lr=1e-4)
learn.fine_tune(args.epochs, lr=1e-4)

# Save the trained model in the outputs folder
os.makedirs('outputs', exist_ok=True)
torch.save(model.state_dict(), 'outputs/quiddler.pt')

# Test the model in the quiddler game context
def get_cards(pred, n=100):
    raw_scores = pred['scores']
    best_scores = raw_scores.argsort()[:-(n+1):-1]
    cards = np.array([class_map.get_id(id) for id in pred['labels']])[best_scores]
    scores = np.array([str(int(sc)) for sc in np.floor(raw_scores*100)])[best_scores]
    mean_score = np.mean(raw_scores[best_scores])
    min_score = np.min(raw_scores[best_scores])
    b_ord = np.array([bb.xyxy[0] for bb in pred['bboxes']])[best_scores].argsort()
    return ','.join(cards[b_ord]), ','.join(scores[b_ord]), mean_score, min_score

def predict_cards(fname, cards):
    img = open_img(fname)
    infer_ds = Dataset.from_images([img], valid_tfms)
    batch, samples = faster_rcnn.build_infer_batch(infer_ds)
    preds = faster_rcnn.predict(model=model, batch=batch, detection_threshold=0.5)
    # Log this as an image
    f,p = plt.subplots(1,1)
    show_img(draw_pred(samples[0]["img"], preds[0], class_map=class_map, denormalize_fn=denormalize_imagenet), ax=p)
    run.log_image(fname.stem, plot=f)
    n_cards = len(cards.split(','))
    results = get_cards(preds[0], n_cards)
    err = 0
    pred_cards = results[0].split(',')
    for i,c in enumerate(cards.split(',')):
        if i >= len(pred_cards) or pred_cards[i] != c:
            err += 1
    return (*results, err/n_cards)

test_path = Path(data_dir)/'test'
df = pd.read_csv(test_path/'testcards.csv')
df['preds'],df['probs'],df['mean'],df['min'],df['err'] = zip(*df.apply(lambda r: predict_cards(test_path/r.fname, r.cards), axis=1))
run.log_table('Card hand recognition results', df.to_dict(orient='list'))

run.log('clf_mean',np.mean(df['mean'])) # <-- Maximize this
run.log('clf_min',np.mean(df['min'])) # <-- Maximize this
run.log('clf_error',sum(df['err'])) # <-- Minimize this

run.complete()
