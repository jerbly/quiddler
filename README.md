### Quiddler
Quiddler is a simple card game where you compete to make high scoring words from your hand of cards. Imagine Scrabble but as a card game.

This project is really an excuse to learn some ML and CV. It's puposefully an end-to-end project for that reason. From taking the photos and data labelling through training and tuning to inference and ultimately the full game.

As of today, 27th November 2020, the project is not complete but getting close so I'm pushing this to share.

![Moving cards](images/cards.gif)

The gif above doesn't really do this justice. This actually runs at around 25fps. The point of this webcam script though was to practically see how well the model was trained for changes in orientation and size. In the real game setting there will be fixed cameras and the cards will be placed on a flat surface.

### VGG Image Annotator (VIA)
I used [VIA](http://www.robots.ox.ac.uk/~vgg/software/via/) to label 120 photos taken with an iPhone or webcam.

![VIA](images/via.png)

### IceVision and fastai
The object detection here is provided by the awesome [IceVision](https://airctic.com/) project in [fastai](https://docs.fast.ai/) mode. To get to this level of performance I have not needed to do very much tweaking beyond picking appropriate training augmentations. 

### AzureML
I happened to also be learning AzureML so I threw this into the mix too. There a couple of notebooks here that use the AzureML SDK to run training and inference services.

![Training](images/azureml1.png)

A simple callback in fastai is used to send metrics to the AzureML Workspace. Also, images and tables of results can be logged as part of the run:

![Logged Image](images/azureml2.png)

And, of course, the Quiddler dataset itself is stored and registered in the workspace:

![Dataset](images/azureml3.png)

### The Game
Coming soon...