import requests
import streamlit as st


API_URL = "http://127.0.0.1:8000/predict"


st.set_page_config(
    page_title="Face Gender Classification",
    page_icon="👤",
    layout="centered"
)


st.title("Face Gender Classification")

st.write(
    "Upload an image and the trained ResNet18 model "
    "will classify the image."
)


uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is not None:

    st.image(
        uploaded_file,
        caption="Uploaded Image",
        use_container_width=True
    )

    if st.button("Predict"):

        with st.spinner("Running prediction..."):

            try:

                files = {
                    "file": (
                        uploaded_file.name,
                        uploaded_file.getvalue(),
                        uploaded_file.type
                    )
                }

                response = requests.post(
                    API_URL,
                    files=files
                )

                response.raise_for_status()

                result = response.json()

                prediction = result["prediction"]
                confidence = result["confidence"]

                st.success(
                    f"Prediction: {prediction.upper()}"
                )

                st.metric(
                    "Model Confidence",
                    f"{confidence * 100:.2f}%"
                )

            except requests.exceptions.ConnectionError:

                st.error(
                    "Could not connect to the FastAPI server. "
                    "Make sure the API is running."
                )

            except Exception as error:

                st.error(
                    f"Something went wrong: {error}"
                )