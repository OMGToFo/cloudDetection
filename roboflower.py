import streamlit as st
from PIL import Image


import pandas as pd


import tempfile


import os
from dotenv import load_dotenv
load_dotenv(".env")

roboflowApi = os.getenv("roboflowApi")



from roboflow import Roboflow
rf = Roboflow(api_key=roboflowApi)
project = rf.workspace().project("clouds-hfkdk")
model = project.version("1").model

st.title("Simple Cloud Detection")
st.info("First try of Image Detection with Roboflow. So far only trained with 50 Images")
    
# File upload
#st.write('Select an image to upload.')
uploaded_file = st.file_uploader('Upload an image',
                                         type=['png', 'jpg', 'jpeg'],
                                         accept_multiple_files=False)


## Add in sliders.
#confidence_threshold = st.sidebar.slider("Confidence threshold (%): What is the minimum acceptable confidence level for displaying a bounding box?", 0, 100, 40, 1)
#overlap_threshold = st.sidebar.slider("Overlap threshold (%): What is the maximum amount of overlap permitted between visible bounding boxes?", 0, 100, 30, 1)



    
if uploaded_file is not None:


        # Save the uploaded image to a temporary file
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                tmp_filename = tmp_file.name
                tmp_file.write(uploaded_file.read())
        #ladda image from temp location
        img = Image.open(tmp_filename)


        predictions = model.predict(tmp_filename) #, overlap=30,confidence=40)

        predictions_json = predictions.json()

        #jsonExpander = st.expander("Show all Info")
        #with jsonExpander:
        #        st.write(predictions_json)





        # Create DataFrame from predictions
        predictions_data = predictions_json["predictions"][0]["predictions"]
        df = pd.DataFrame(predictions_data).transpose()
        df = df.rename(columns={"confidence": "Confidence"})
        df.index.name = "Cloud Type"

        # Sort DataFrame by Confidence in descending order
        df_sorted = df.sort_values(by="Confidence", ascending=False)
        bestPrediction = df_sorted.index[0]
        
        #st.info(highestCoicidenceClass)

        _="""
        predictedClassFound = (predictions_json["predictions"][0]["predicted_classes"])
        st.write("predictedClassFound: ",predictedClassFound)
        if predictedClassFound == []:
                st.write("no class found")
        else:
               st.write(" class found") 

        # Extract highest predicted class
        predicted_classes = predictions_json["predictions"][0]["predicted_classes"][0]

        # Display predicted classe in Streamlit
        if predicted_classes !="":
                st.subheader("Predicted Cloud Type: " + predicted_classes)
     
                
        """
        if len(df_sorted)>0:
                # Display only the first row in Streamlit
                st.subheader("Prediction: " + bestPrediction)
                #st.write(f"Predicted Cloud Type: {df_sorted.index[0]}")
                st.write(f"Confidence: {df_sorted['Confidence'].iloc[0]:.2%}")
                  

        if bestPrediction =="Cirrus":
                st.info("Cirrus Clouds: High-altitude clouds that are wispy and thin. They often indicate fair weather but can also precede a change in the weather.")


        if bestPrediction =="Cumulus":
                st.info("Puffy, white clouds with a flat base. They are often associated with fair weather, but larger cumulus clouds can develop into storm clouds.")


        if bestPrediction =="Stratus":
                st.info("Stratus Clouds: Low-altitude clouds that form a continuous layer, covering the sky like a blanket. They often bring overcast skies and light precipitation.")


        if bestPrediction =="Nimbostratus":
                st.info("Nimbostratus Clouds: Thick, dark clouds that cover the sky and bring continuous, steady precipitation.")



        if bestPrediction =="Altostratus":
                st.info("Altostratus Clouds: Mid-level clouds that often cover the entire sky, creating a diffuse veil. They can precede storms with continuous rain or snow.")


        if bestPrediction =="Altocumulus":
                st.info("Altocumulus Clouds: Mid-level clouds that appear as white or gray patches, often forming a layer. They don't usually bring precipitation but can signal a change in the weather.")



        if bestPrediction =="Stratocumulus":
                st.info("Stratocumulus Clouds: Low, lumpy clouds that cover the sky without a distinct shape. They may bring light precipitation.")


        if bestPrediction =="Cumulonimbus":
                st.info("Cumulonimbus Clouds: Towering clouds that can reach the stratosphere. They often bring thunderstorms, heavy rain, lightning, and other severe weather.")





        st.image(img, caption='Uploaded Image.', use_column_width=True)

        st.divider()



        # Display DataFrame in Streamlit
        st.subheader("Prediction Probabilities")
        st.dataframe(df_sorted)

# visualize your prediction
# model.predict("YOUR_IMAGE.jpg", confidence=40, overlap=30).save("prediction.jpg")

# infer on an image hosted elsewhere
# print(model.predict("URL_OF_YOUR_IMAGE", hosted=True, confidence=40, overlap=30).json())


