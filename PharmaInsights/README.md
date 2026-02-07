# PharmaInsights – AI Powered Drug Safety Guide

PharmaInsights is an AI-based web application that analyzes medicine ingredient information and predicts the **drug risk level** such as **Safe, Moderate, or High**. The system combines OCR, basic NLP techniques, and a Machine Learning model to help users understand potential risks and side effects of medicines.

---

## 🚀 Features

* Extracts medicine ingredients from images using **Tesseract OCR**
* Cleans and processes raw text using **basic NLP techniques**
* Identifies active ingredients from extracted text
* Predicts drug risk level using a **Logistic Regression model**
* Displays risk level and side effects on a web interface
* Simple **Flask backend** connecting ML model and frontend

---

## 🧠 How the System Works

1. **OCR (Optical Character Recognition)**

   * Users upload or provide medicine information
   * Tesseract OCR extracts raw text from medicine labels

2. **NLP (Text Processing)**

   * Cleans OCR output by removing noise and unnecessary symbols
   * Tokenizes text to extract ingredient names
   * Matches extracted ingredients with known ingredients in the dataset
   * Converts unstructured text into structured input for the ML model

3. **Machine Learning Model**

   * Uses **Logistic Regression** trained on a small, structured dataset
   * Predicts drug risk level: Safe / Moderate / High

4. **Backend & Frontend Integration**

   * Flask handles API requests and model inference
   * Frontend (HTML, CSS, JavaScript / basic React components) displays results

---

## 🛠️ Tech Stack

### Backend

* Python
* Flask
* Pandas
* Scikit-learn

### Machine Learning

* Logistic Regression
* Feature extraction from ingredient data

### OCR & NLP

* Tesseract OCR (pytesseract)
* OpenCV (image preprocessing)
* Rule-based NLP (text cleaning, tokenization, ingredient matching)

### Frontend

* HTML
* CSS
* JavaScript
* Basic React components

---

## 📌 Project Structure

* `app.py` – Flask application for API handling
* `train_model.py` – Model training and saving
* `ocr.py` – OCR and text extraction logic
* `frontend/` – UI files to display results
* `requirements.txt` – Project dependencies

---

## 🎯 Conclusion

PharmaInsights demonstrates how **OCR, basic NLP, and Machine Learning** can be combined to build a practical healthcare-focused application. The project focuses on clarity, simplicity, and real-world applicability using a lightweight ML model and clean backend–frontend integration.

---

## 📚 Future Enhancements

* Larger and more diverse dataset
* Advanced NLP techniques
* Deep learning-based models
* Mobile-friendly UI
* Drug interaction analysis
