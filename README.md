💊 PharmaInsights – AI-Powered Drug Safety Guide
🚀 Overview

PharmaInsights is an AI-powered web application designed to analyze pharmaceutical drug compositions and assess potential safety risks.
The system helps users understand whether a medicine is Safe, Moderate Risk, or High Risk based on ingredient composition, dosage levels, and known side effects.

This project aims to promote drug safety awareness by leveraging Machine Learning, NLP, and OCR techniques.

🎯 Problem Statement

Many people consume medicines without fully understanding:

Ingredient composition

Dosage limits

Possible side effects

Drug safety risks

Existing drug information is often complex and not user-friendly.
PharmaInsights simplifies this process by providing AI-based risk analysis in an easy-to-understand format.

💡 Solution

PharmaInsights allows users to:

Upload or enter medicine ingredient details

Analyze drug composition using AI models

Get instant safety classification and warnings

View possible side effects and precautions

🧠 How It Works

Input Collection

Medicine ingredients and composition (text/image)

OCR extracts text from medicine labels using Tesseract OCR

Data Preprocessing

Cleaning ingredient names

Normalizing dosage values

Feature encoding for ML model

Machine Learning Model

Classification model trained on drug composition data

Predicts safety level:

🟢 Safe

🟡 Moderate Risk

🔴 High Risk

Backend Processing

Flask API handles requests

Model inference and result generation

Output Display

Risk level

Warnings if dosage exceeds normal limits

Possible side effects

🛠️ Technologies Used
🔹 Frontend

HTML

CSS

JavaScript

Bootstrap

🔹 Backend

Python

Flask (REST API)

🔹 Machine Learning

Scikit-learn

NumPy

Pandas

🔹 OCR & NLP

Tesseract OCR

Text preprocessing techniques

🔹 Tools

Git & GitHub

VS Code

Anaconda

📊 Input & Output Features
✅ Input Features

Drug name

Ingredient names

Ingredient composition values

Medicine label image (optional)

📤 Output Features

Safety classification (Safe / Moderate Risk / High Risk)

Dosage warning alerts

Possible side effects

Precaution recommendations

📈 Model Evaluation Metrics

Accuracy

Precision

Recall

F1-Score

Confusion Matrix

These metrics ensure reliable and consistent predictions.

🌟 Key Features

AI-based drug safety prediction

OCR support for medicine labels

User-friendly risk classification

Real-time analysis

Educational and awareness-driven design

🔮 Future Enhancements

Integration with real-time drug databases

Support for multiple languages

Mobile application version

Advanced deep learning models

Doctor/Pharmacist recommendation system
🧪 Project Use Cases

Patients and general users

Medical students

Pharmacists

Healthcare awareness platforms
