**SymptoCare – Symptom-Based Disease Prediction Desktop App**
=============================================================

**SymptoCare** is a Python desktop application that predicts likely diseases from patient symptoms and demographic details.It combines a **scikit-learn machine learning model** with a **modern Tkinter GUI**, interactive dashboards, and detailed prediction-history analytics.

⭐ **Project Features**
----------------------

### 🧠 **Machine Learning**

*   Multi-class disease prediction using a **RandomForestClassifier**.
    
*   Custom transformer for:
    
    *   Age
        
    *   Gender
        
    *   Symptom count
        
    *   Multi-hot encoded symptoms
        
*   Entire pipeline saved as: **models/disease\_model.joblib**
    

🖥️ **Desktop GUI (Tkinter)**
-----------------------------

### 🔐 **Login Screen**

*   **Username:** admin
    
*   **Password:** admin
    

### 📊 **Main Dashboard**

*   Sidebar navigation:
    
    *   Dashboard
        
    *   Predict
        
    *   Analytics
        
    *   History
        
*   Top summary cards:
    
    *   Total patients
        
    *   Number of diseases
        
    *   Model accuracy
        
*   Embedded chart:
    
    *   **Patients per Disease** (disease\_counts.png)
        

### 🩺 **Predict Tab**

Inputs:

*   Age (numeric)
    
*   Gender (Male / Female / Other)
    
*   Symptoms (comma-separated, e.g., fever, cough, fatigue)
    

Outputs:

*   Top predicted disease with probability
    
*   **Top 3 diseases** with individual probabilities
    

⚠️ _Safety note: Not a medical diagnosis — for educational use only._

### 📈 **Analytics Tab**

*   Reads **prediction\_history.csv**
    
*   Generates a **“Top Predicted Diseases”** bar chart
    
*   Shows how the app has been used over time
    

### 📜 **History Tab**

Displays a table with:

*   Timestamp
    
*   Age, gender
    
*   Symptoms (truncated)
    
*   Predicted disease
    
*   Top probability (%)
    

Buttons:

*   **Refresh** – reload history
    
*   **Clear History** – deletes the CSV and resets charts
    

⚙️ **Source Files Overview**
----------------------------

### 🔧 **preprocessing.py**

Defines **FullFeatureTransformer**, which prepares:

*   Age
    
*   Gender
    
*   Symptom\_Count
    
*   Multi-hot encoded Symptoms
    

### 🤖 **train\_model.py**

*   Loads Healthcare.csv
    
*   Trains a RandomForest classifier using the custom transformer
    
*   Saves:
    
    *   disease\_model.joblib
        
    *   Metrics (metrics.txt)
        

### 📊 **eda.py**

Generates dataset-level charts:

*   Patients per disease
    
*   Age distribution
    
*   Top symptoms
    

Outputs saved to: reports/eda\_plots/

### 🖥️ **gui.py**

Implements the full GUI:

*   Login window
    
*   Dashboard
    
*   Prediction form
    
*   Analytics chart rendering
    
*   History table
    

Includes helper:**append\_prediction()** – logs each prediction into prediction\_history.csv.

🚀 **How It Works (High-Level)**
--------------------------------

### **1️⃣ Train the model**

`   python -m src.train_model   `

Creates:

*   models/disease\_model.joblib
    
*   reports/metrics.txt
    

### **2️⃣ Generate EDA Charts**

`   python -m src.eda   `

### **3️⃣ Launch the Application**

`   python -m src.gui   `

Login with:

Username:admin
Password:admin

### **4️⃣ Making Predictions**

*   Enter age
    
*   Select gender
    
*   Enter symptoms
    
*   Click **Predict Disease**
    

Application will:

*   Create a one-row DataFrame
    
*   Run predict\_proba
    
*   Display top 3 diseases
    
*   Append logs to prediction\_history.csv
    

### **5️⃣ Viewing Analytics & History**

*   **Analytics tab:** Shows usage-based trends
    
*   **History tab:** Full table of past predictions
    
*   **Clear History:** empty the CSV
    

🛠️ **Setup Instructions**
--------------------------

### 1\. Install Python 3.9+

Download from [https://python.org](https://python.org)

### 2\. Create Virtual Environment

`   python -m venv venv   `

Activate:

**Windows:**

`   venv\Scripts\activate   `

**Linux/macOS:**

`   source venv/bin/activate   `

### 3\. Install Dependencies

`   pip install numpy pandas scikit-learn seaborn matplotlib pillow   `

### 4\. Place Dataset

Put **Healthcare.csv** in the data/ folder.

### 5\. Run Training & EDA

`   python -m src.train_model  python -m src.eda   `

### 6\. Run the App

`   python -m src.gui   `

📌 **Note**
-----------


This project is for **educational use only** and should not be used for real medical diagnosis.


