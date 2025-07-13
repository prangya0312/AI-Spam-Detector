# 📩 AI Spam Detector (Streamlit App)

A simple and effective AI-powered web app built using **Streamlit** that detects **spam or fraudulent SMS messages** using Natural Language Processing (NLP) and Machine Learning.

This project supports:
- ✅ Manual SMS input
- ✅ Bulk spam detection via `.csv` or `.pdf` upload
- ✅ Downloadable prediction results
- ✅ Model persistence using `joblib`

---

## 🚀 Live Demo

👉 [Click to Use the App](https://prangya0312-ai-spam-detector.streamlit.app/)

---

## 📂 Folder Structure

AI-Spam-Detector/
├── app.py # Streamlit web app
├── spam_model.pkl # Trained spam detection model
├── vectorizer.pkl # TF-IDF vectorizer
├── requirements.txt # Python dependencies
├── README.md # Project documentation
└── spam.csv # Dataset used for training (optional)


---

## 📥 Features

- 🔍 **Single Message Detection**  
  Type any message and instantly check if it's spam or not.

- 📂 **Bulk Detection**  
  Upload a `.csv` or `.pdf` file of messages. The app processes each one and tells you if it's spam.

- 📤 **Download Results**  
  Export prediction results as CSV.

---

## 📊 Technologies Used

- Python 3
- Streamlit
- scikit-learn
- Pandas
- Joblib
- PyMuPDF (`fitz` for PDF reading)
- Natural Language Toolkit (NLTK)

---

## ⚙️ Run Locally

1. Clone the repo:

   ```bash
   git clone https://github.com/prangya0312/ai-spam-detector.git
   cd ai-spam-detector
pip install -r requirements.txt

streamlit run app.py

🧠 Model Details
Algorithm: Multinomial Naive Bayes

Trained on: SMS Spam Collection Dataset

Text Processing: TF-IDF vectorization

🛡️ Future Enhancements
Add login/authentication

Store analysis history per user

REST API version

Mobile UI optimization

📜 License
This project is open-source and available under the MIT License.

🙋‍♀️ Developed By
Prangya Gantayat
B.Tech (CSE) | SOA University
📍 Odisha, India




