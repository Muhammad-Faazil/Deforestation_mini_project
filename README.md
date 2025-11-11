# 🌲 Deforestation Detection using CNN

A deep learning project that detects deforestation areas in satellite and aerial images using **Convolutional Neural Networks (CNNs)**.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28-red)
![Accuracy](https://img.shields.io/badge/Accuracy-88.5%25-green)

---

## 🚀 Features

- 🌍 **Image Classification:** Detects *deforestation* vs *non-deforestation* areas.  
- 💻 **Web Interface:** Streamlit-based web app for an easy-to-use GUI.  
- 🎯 **High Accuracy:** Achieves **88.5% test accuracy**.  
- 📊 **Confidence Scoring:** Displays prediction probabilities.  
- 🗂️ **Batch Processing:** Supports multiple images at once.  
- 🕒 **Prediction History:** Keeps logs with timestamps.  
- 📥 **Export Results:** Download results as a CSV file.

---

## 📸 Demo

> *(Add a screenshot or screen recording here later)*

![App Screenshot](demo.png)

---

## 🛠️ Installation

### ✅ Prerequisites
- Python **3.8+**
- **pip** (Python package manager)

### ⚙️ Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/Muhammad-Faazil/Deforestation_mini_project.git
   cd Deforestation_mini_project
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   ```

3. **Activate the environment**
   - **Windows:**
     ```bash
     venv\Scripts\activate
     ```
   - **Linux/Mac:**
     ```bash
     source venv/bin/activate
     ```

4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

5. **Run the app**
   ```bash
   streamlit run app.py
   ```

Once the app starts, open your browser and go to:  
👉 [http://localhost:8501](http://localhost:8501)

---

## 📂 Project Structure

```
Deforestation_mini_project/
│
├── app.py                 # Streamlit web app
├── model/
│   └── deforestation_model.keras
├── data/
│   ├── train/
│   └── test/
├── utils/
│   └── preprocessing.py
├── requirements.txt
├── demo.png
└── README.md
```

---

## 🧠 Model Overview

- **Architecture:** Custom CNN (Conv2D → MaxPool → Flatten → Dense)
- **Optimizer:** Adam
- **Loss Function:** Categorical Crossentropy
- **Metrics:** Accuracy
- **Dataset:** Satellite imagery (deforestation vs. forested regions)

---

## 📊 Results

| Metric | Value |
|--------|--------|
| Accuracy | **88.5%** |
| Validation Loss | 0.29 |
| Precision | 0.87 |
| Recall | 0.88 |

---

## 💡 Future Enhancements

- 🔍 Integrate Grad-CAM for visual explanation of predictions  
- ☁️ Deploy on AWS or Hugging Face Spaces  
- 📱 Build a mobile version using TensorFlow Lite  

---

## 🧑‍💻 Author

**Muhammad Faazil Abbas**  
🌐 [GitHub Profile](https://github.com/Muhammad-Faazil)

---

## 🪪 License
This project is licensed under the **MIT License** – see the [LICENSE](LICENSE) file for details.
