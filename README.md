🧠 Brain Stroke Detection

 📌 Project Overview
This project is a Deep Learning-based web application that detects the presence of brain stroke from MRI scan images. The system allows users to upload brain MRI images and predicts whether a stroke is detected or not using a trained neural network model.

The application is built using Flask for the backend and a trained `.h5` deep learning model for prediction.

 🎯 Objective
To assist in early detection of brain stroke using image-based classification through a web interfac

 🧠 Model Details
- Trained Deep Learning model (.h5 file)
- Image-based classification
- Model integrated with Flask backend for real-time prediction

---

 🛠 Technologies Used

 ##  Backend
- Python
- Flask
- TensorFlow / Keras
- NumPy

### Frontend:
- HTML
- CSS
- JavaScript

---

## 📁 Project Structure
brain_stroke_detection/
|
|-- backend/
|   |-- app.py
|   |-- brain_stroke_detection_model.h5
|   |-- requirements.txt
|
|-- static/
|   |-- css/
|   |-- js/
|   |-- uploads/
|
|-- templates/
|   |-- index.html
|
|-- app.py
🚀 How to Run the Project
✔ https://github.com/hemachandrika04/brain_stroke_detection.git

✔ cd brain_stroke_detection/backend

✔ pip install -r requirements.txt

✔ python app.py

✔ http://127.0.0.1:5000/
## 📊 Output

After uploading a brain MRI image:

✔ The system processes the image  
✔ The trained deep learning model performs prediction  
✔ The result is displayed on the web page  

### Possible Outputs:

- 🟢 **No Stroke Detected**
- 🔴 **Stroke Detected**

The prediction result is displayed clearly on the interface along with the uploaded image.

## 🔮 Future Improvements

- Improve model accuracy
- Add patient data logging
- Deploy to cloud (Render / AWS)
- Add confidence score display
- Add multiple disease classification







