import streamlit as st

import tensorflow as tf

from tensorflow.keras.preprocessing import image

import numpy as np


# LOAD MODEL
model = tf.keras.models.load_model("Rice_Disease_Model.h5")


# CLASS NAMES
class_names = [

    "Bacterialblight",

    "Brownspot",

    "Leafsmut"

]


# TITLE
st.title("Rice Leaf Disease Detection")


# IMAGE UPLOAD
uploaded_file = st.file_uploader(

    "Upload Rice Leaf Image",

    type=["jpg", "png", "jpeg"]

)


if uploaded_file is not None:

    # LOAD IMAGE
    img = image.load_img(

        uploaded_file,

        target_size=(128,128)

    )

    # PREPROCESS
    img_array = image.img_to_array(img)

    img_array = img_array / 255.0

    img_array = np.expand_dims(img_array, axis=0)


    # PREDICTION
    prediction = model.predict(img_array)

    ind = np.argmax(prediction[0])

    confidence = np.max(prediction[0]) * 100


    # SHOW IMAGE
    st.image(

        uploaded_file,

        caption="Uploaded Image",

        use_container_width=True

    )


    # RESULT
    st.success(

        f"Predicted Disease: {class_names[ind]}"

    )

    st.write(

        f"Confidence: {confidence:.2f}%"

    )