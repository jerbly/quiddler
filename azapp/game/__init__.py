import logging
import azure.functions as func
from .quiddler_game import Quiddler
from pathlib import Path
import json

quiddler = None

def main(req: func.HttpRequest, context: func.Context) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')
    global quiddler

    if not quiddler:
        logging.info(f'Context function directory: {context.function_directory}')
        path = Path(context.function_directory) / 'sowpods.txt'
        quiddler = Quiddler(vocab_path=path)

    hand = req.params.get('hand')
    deck = req.params.get('deck')
    if not (deck and hand):
        try:
            req_body = req.get_json()
        except ValueError:
            pass
        else:
            hand = req_body.get('hand')
            deck = req_body.get('deck')

    if deck and hand:
        best_play = quiddler.get_best_play(hand.split('/'),deck)
        return func.HttpResponse(json.dumps(best_play))
    else:
        return func.HttpResponse(
             "Hand and Deck required",
             status_code=400
        )
