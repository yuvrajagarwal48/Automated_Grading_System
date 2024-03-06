import numpy as np
import tensorflow as tf
import string
import nltk 
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from tensorflow import keras
import transformers
from transformers import DistilBertTokenizer, TFDistilBertModel
import keras_ocr
import flask

def image_to_text(img_path):
    # Initialize the OCR pipeline
    pipeline = keras_ocr.pipeline.Pipeline()

    # Read the image
    image = keras_ocr.tools.read(img_path)
    
    # Perform OCR on the image
    predictions = pipeline.recognize([image])

# Extract and print the text from the predictions
    text=''
    for text_result in predictions[0]:
        text=text+text_result[0]+" "
    return text

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

stopwords = nltk.corpus.stopwords.words('english')
lemmatizer = WordNetLemmatizer()

custom_objects = {
    'DistilBertTokenizer': DistilBertTokenizer,
    'TFDistilBertModel': TFDistilBertModel
}
def clean_text(text):
    text=str(text)
    tokens = word_tokenize(text)
    tokens[0]=tokens[0][2:]
    cleaned_text=[lemmatizer.lemmatize(word.lower()) for word in tokens if word.lower() not in stopwords and word not in string.punctuation]
    return ' '.join(cleaned_text)

tokenizer_bert = transformers.DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model=keras.models.load_model('best_model.h5',custom_objects=custom_objects)

def predict(text):
    tokenized_text=tokenizer_bert(clean_text(text), padding='max_length', max_length = 512, truncation=True)['input_ids']
    prediction=model.predict(tokenized_text)
    print(np.argmax(prediction[0]))
    return np.argmax(prediction[0])


text='''
All boys putting up their suggestions to make their night more of fun and memorable. Rohan suggested “Let’s go to the cinema and watch the latest movie ”. Rohit interrupted “No, No i dont want to waste this night by sitting on a chair for 3 hours and watching that romantic film” . Aryan then commented “ How about watching a thriller in the room itself.” Rohit agreed by telling that he wont mind watching a thriller .Mohit and Chirag were still not satisfied and were constantly giving a bored look to each other. Then suddenly there was a brightening sparkle on Mohit’s face . Mohit discussed his plan with Chirag and then both came up with a suggestion “ We both will agree to join you guys on 1 condition that if we could hangout that night somewhere outside the room after we are done with the thriller .Rohit , Rohan and Aryan agreed with no option left with them.

Finally it was 9 pm , all of the boys were excited ,turning off all the lights , drawing off all the curtains and all took their seats with a bowl of popcorn. The horror film started and all of them staring at the screen , seemed as if their eyeballs would pop out. All of them took a deep breath during the 2 minute interval in the movie. Rohit went to the kitchen to drink water and suddenly he heard a voice from the bathroom , opposite the kitchen as if someone had left the tap open and a bucket under it. The sound was turning out to be shriller and shriller so Rohit rushed to the bathroom to close the tap and guess what he saw ? He saw that all taps were closed and all the buckets were put upside down . This worried Rohit , he called all his friends and told them about this incident but no on believed him . Rohan said “ the horror film has got into your head , its better you go to sleep now”. Mohit and Chirag giggled. Aryan did not speak anything at the moment as he was not able to jump to any conclusions at the moment.
'''
































print(predict(text))


# def pipeline(data,flag):
#     text=''
#     if(flag==0):        #image
#         text=image_to_text(data)
#     else:
#         text=clean_text(data)
#     label=predict(text)
#     return label
    
        
# print(pipeline('image.jpg',0))