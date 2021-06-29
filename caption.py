import matplotlib.pyplot as plt
import pickle
import numpy as np
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model, Model
import warnings
warnings.filterwarnings("ignore")

my_model = load_model("./model_weights/my_model.h5")

temporary_model = ResNet50(weights="imagenet", input_shape=(224,224,3))
resnet_model = Model(temporary_model.input, temporary_model.layers[-2].output)

with open("./storage/i2w.pkl", "rb") as i2w:
    idx2word = pickle.load(i2w)  
    maximum_length = 35

with open("./storage/w2i.pkl", "rb") as w2i:
    word2idx = pickle.load(w2i)

def image_preprocess(img):
    img = image.load_img(img, target_size=(224,224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img

def encode_image_resnet(img):
    img = image_preprocess(img)
    img_feature_vector = resnet_model.predict(img)
    img_feature_vector = img_feature_vector.reshape(1, img_feature_vector.shape[1])
    return img_feature_vector

def predict_caption(input_image_for_model):
    in_text = "startseq"
    
    for i in range(maximum_length):
        sequence = [word2idx[w] for w in in_text.split() if w in word2idx]
        sequence = pad_sequences([sequence], maxlen=maximum_length, padding='post')
        y_prediction =  my_model.predict([input_image_for_model,sequence])
        y_prediction = y_prediction.argmax()
        word = idx2word[y_prediction]
        in_text+= ' ' +word
        if word =='endseq':
            break
            
    predicted_caption =  in_text.split()
    predicted_caption = predicted_caption[1:-1]
    predicted_caption = ' '.join(predicted_caption)
    return predicted_caption

def caption_image(input_img): 
    input_image_for_model = encode_image_resnet(input_img)
    caption = predict_caption(input_image_for_model)
    return caption