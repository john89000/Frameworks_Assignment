# CORD-19 Research Challenge – Data Exploration

This project explores the **CORD-19 dataset** (COVID-19 Open Research Dataset) using Python, Pandas, and visualization tools. A simple Streamlit web application is included to interactively view the findings.

---

## 📂 Dataset
You can download the dataset from Kaggle:  
👉 [CORD-19 Research Challenge](https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge)

⚠️ **Note:** The full dataset is very large. For this project, you only use the **`metadata.csv`** file or a small sample of the data.

---

## 🛠️ Required Tools
- **Python 3.7+**
- **pandas** → data manipulation
- **matplotlib / seaborn** → visualization
- **Streamlit** → web application
- **Jupyter Notebook** (optional, for exploration)

---

## 📦 Installation
Clone the repository:

```bash
git clone https://github.com/john89000/Frameworks_Assignment.git
cd Frameworks_Assignment

Install the required Python packages:

pip install pandas matplotlib seaborn streamlit

▶️ Usage
1. Run Jupyter Notebook (optional)

If you want to explore the dataset interactively:

jupyter notebook

2. Run the Streamlit app

To launch the web application:

streamlit run app.py


This will start a local server and open the app in your browser at:
👉 http://localhost:8501

📊 Features

Load and preview the CORD-19 metadata.csv file

Generate visualizations to show trends and patterns in the dataset

Explore data interactively with Streamlit
