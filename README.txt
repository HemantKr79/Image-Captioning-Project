'#' <h1> Image Caption Generator </h1>

Problem Statement:
We aim to build an Image Captioning Bot using machine learning algorithms and designing a user interface for the user to get caption for the image..

What is Image Captioning?
Image Captioning is a difficult task of producing correct textual descriptions of an image in natural language with context related to the image.

ML Model:
The model made in this project will take picture as input and predict the caption for image. The model is based on encoder-decoder architecture. A two-dimensional convolutional neural network is used to encode the image features, and a one-dimensional convolutional neural network is used to encode the word sequences of the caption data. Later, the decoder predict the caption in a word by word manner using encoded image and text features are merged and passed to it. The model was integrated with the web using flask framework using HTML and bootstrap.

Programming Languages : Python, HTML

Modules Used:
Pandas
Numpy
Tensorflow
Nltk
Pickle
Json
