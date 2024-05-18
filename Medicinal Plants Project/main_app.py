# Library imports
import numpy as np
import streamlit as st
import cv2
from keras.models import load_model
import tensorflow as tf

# Loading the Model
model = load_model('medicinal_plants_cnn.h5')
                    
# Name of Classes
CLASS_NAMES = ('Abelmoschus moschatus medik(Ambrette )', 'Aloe vera (L.) Burm.F(Aloe Vera)', 'Hibiscus rosasinensis(Red Hibiscus)', 'Kaempferia Galanga(Aromatic ginger)', 'Kalanchoe Pinnata (Lam.) Pers(Miracle leaf)', 'Lasia Spinosa (L.) Thwaites(Lesia)', 'Lawsonia inermis L.(Henna)', 'Leucas aspera Link(Thumba)', 'Mentha arvensis L(Corn Mint)', 'Mesua ferrea L.(Nagakesar)', 'Mimusops elengi L.(Spanish cherry)', 'Nyctanthes arbor-Tristis L.(Night Blooming Jasmine)', 'Psidium guajava L.(Guava Seed)', 'Rauvolfia serpentina Benth. Ex Kurz(Serpentine root)', 'Rotheca serrata (L.) Steane & Mabb.(Clerodendrum, Bharangi)', 'Vanilla planifolia(Flat-leaved vanilla)', 'Vitex negundo L.(Chinese Chaste Tree)', 'Zanthoxylum nitidum DC.(Shiny-leaf prickly-ash)', 'Zingiber officinale Rosc.(Ginger rhizome)', 'Ziziphus Jujuba Mill.(Jujube)')

# Setting Title of App
st.title("Plant Detection")
st.markdown("Upload an image of the plant leaf")

# Uploading the dog image
plant_image = st.file_uploader("Choose an image...", type = "jpg")
submit = st.button('PREDICT PLANT')

# On predict button click
if submit:
    if plant_image is not None:
        # Convert the file to an opencv image.
        file_bytes = np.asarray (bytearray(plant_image.read()), dtype = np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)
        
        # Displaying the image
        st.image(opencv_image, channels="BGR")
        st.write(opencv_image.shape)
        
        # Resizing the image
        opencv_image = cv2.resize(opencv_image, (150, 150))
        
        # Convert image to 4 Dimension
        opencv_image.shape = (1, 150, 150, 3)
        
        #Make Prediction
        Y_pred = model.predict(opencv_image)
        result = CLASS_NAMES[np.argmax(Y_pred)]
        st.title(str("This medicinal plant is " +  result))
