# 💊 PharmaInsights: AI-Powered Drug Safety Guide

PharmaInsights is an AI-based web application designed to help users understand whether a medicine is safe to use or not, especially for rural areas and elderly patients who may not be able to interpret complex medicine labels. This tool analyzes the ingredients of a drug and predicts its risk level — **Safe**, **Moderate**, or **High** — and also displays possible side effects.

---

## 🚀 Features

- Upload medicine strip images to analyze content
- OCR-based extraction of medicine composition using **Tesseract**
- NLP-based ingredient extraction
- Machine Learning model to predict **risk level**
- Displays possible **side effects** and alerts for overdose

---

## 🧠 Technologies Used

- **Python**, **Flask**
- **Tesseract OCR**
- **OpenCV** for image preprocessing
- **Scikit-learn**, **Pandas**, **NumPy**
- **HTML**, **CSS**, **Bootstrap** for frontend

---

## 🏗️ How It Works

1. User uploads an image of a medicine strip.
2. OCR extracts the text from the image.
3. NLP identifies drug names and dosages.
4. ML model compares data with reference dataset.
5. The system predicts the **risk level** and lists side effects.

---

## 📊 Machine Learning Model

- Collected a dataset of medicines, their compositions, and known side effects.
- Trained classification models like **Logistic Regression** and **Decision Tree**.
- Final model used: **Decision Tree Classifier** (best performance on test data).

---

## ⚠️ Challenges Faced

- Low-quality images: Solved using image preprocessing (grayscale, thresholding).
- Irregular label formats: Handled using regex and flexible parsing.

---

## 📚 Learning & Impact

This project helped me explore the integration of **AI, OCR, NLP, and Web Development**. It’s a real-world application with social impact, especially for non-tech-savvy people or those in rural areas.

---


## 📌 Note

This project was built as part of my academic learning to explore AI in healthcare and is not intended for real medical diagnosis.

---

## 🧑‍💻 Author

Sanapala Dhanalaxmi  
B.Tech CSE (AI & ML), 2026  
Email: dhanalaxmisanapala05@gmail.com

