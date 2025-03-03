Here's your updated GitHub README file, incorporating the latest results while maintaining the requested format.

---

# **🔍 Heart Attack Prediction Using Machine Learning**
### **📊 Predicting Heart Attacks with AI-Driven Models**
![Heart Disease Prediction](https://github.com/user-attachments/assets/ea92c7db-f7ed-4e40-a0c8-cd92fb56cab1)

## **📌 Overview**
Heart disease is one of the leading causes of mortality worldwide, making **early and accurate heart attack prediction crucial** for preventive healthcare. This project applies **machine learning** to analyze structured **health and lifestyle data**, aiming to **identify individuals at higher risk of heart attacks**.

## **🎯 Project Goals**
✅ **Develop & Fine-Tune Machine Learning Models** – Comparing **XGBoost, LightGBM, Logistic Regression, SVM, and Neural Networks** to determine the best predictive model.  
✅ **Optimize Performance through Hyperparameter Tuning** – Using **Grid Search and Cross-Validation** to enhance model accuracy.  
✅ **Feature Engineering for Better Prediction** – Creating new features to **improve recall and precision** in detecting high-risk individuals.  
✅ **Assess Clinical Applicability** – Evaluating models based on **AUC ROC, recall, precision, and medical interpretability** to ensure **practical use in healthcare**.

---

## **🛠️ Tech Stack**
| **Category**  | **Tools & Libraries** |
|--------------|----------------------|
| Programming  | Python |
| Data Science | Pandas, NumPy, Scikit-learn |
| ML Models    | XGBoost, LightGBM, Logistic Regression, SVM, TensorFlow |
| Hyperparameter Tuning | GridSearchCV, Stratified K-Folds |
| Data Visualization | Matplotlib, Seaborn |

---

## **📂 Project Structure**
```plaintext
📁 heart-attack-prediction/
│── 📄 README.md               # Project Documentation
│── 📄 heart_2022_no_nans.csv  # Project Dataset
│── 📄 heart-disease-prediction.ipynb  # Main Notebook with Final Results
```

---

## **🔬 Data Preprocessing & Feature Engineering**
**Data Cleaning & Transformation:**
- **Encoding Categorical Features** – Using **Ordinal Encoding, One-Hot Encoding, and Target Encoding**.
- **Scaling Numerical Features** – Applying **MinMaxScaler** to standardize inputs.
- **Balancing Data** – Using **SMOTE-Tomek and gender-based resampling** to handle **imbalanced data**.
- **Feature Selection** – Using **Feature Importance and Mutual Information** to rank key predictors.
- **Feature Engineering** – Creating **new features** to improve **recall and precision**.

---

## **📊 Machine Learning Pipeline**
1️⃣ **Data Preprocessing** – Cleaning, encoding, and balancing the dataset.  
2️⃣ **Feature Selection & Engineering** – Identifying **key risk factors** for heart attacks.  
3️⃣ **Model Training & Hyperparameter Tuning** – Optimizing ML models using **Grid Search & Cross-Validation**.  
4️⃣ **Performance Evaluation** – **Comparing AUC ROC, recall, precision, and real-world applicability**.  
5️⃣ **Final Model Recommendation** – Selecting the **best model** for practical use in **medical decision-making**.

---

## **🔎 Model Performance Comparison**
| **Model** | **Best ROC AUC (CV)** | **Test ROC AUC** | **Recall (Heart Attack Cases)** | **Precision (Heart Attack Cases)** | **Accuracy** |
|-----------|-----------------|----------------|----------------|----------------|----------------|
| **XGBoost** | 0.899 | **0.857** | 69% | 21% | 84% |
| **LightGBM** | 0.943 | **0.862** | 67% | 26% | 88% |
| **Logistic Regression** | 0.881 | **0.877** | **76%** | **20%** | 82% |
| **Linear SVM** | 0.880 | **0.877** | 53% | 39% | 93% |
| **TensorFlow Neural Network** | 0.879 | **0.879** | 62% | 30% | 90% |
| **Ensemble Model (XGB + LGBM)** | - | **0.863** | 70% | 23% | 86% |

---

## **📌 Key Findings**
✅ **Feature Engineering Improved Prediction** – Adding new **engineered features** significantly **boosted model recall and precision**.  
✅ **Logistic Regression is the Most Reliable Model** – With **76% recall and 20% precision**, it is the **best model for clinical applications**.  
✅ **XGBoost & LightGBM Have Strong Predictive Power** – They achieve **competitive AUC ROC scores but may require calibration** to improve precision.  
✅ **Deep Learning Performed Well but is Less Interpretable** – **TensorFlow’s model reached 0.879 AUC ROC**, but its **lack of transparency limits real-world medical use**.  
✅ **Ensemble Learning Showed Potential** – Combining **XGBoost and LightGBM** in an ensemble model **achieved an ROC AUC of 0.863**, but did not significantly outperform individual models.

---

## **🏆 Final Recommendation**
🔹 **Logistic Regression is the best model** for **heart attack prediction in healthcare**, offering **high sensitivity (recall) while being interpretable** for doctors.  
🔹 **XGBoost and LightGBM** could be **strong alternatives** but may need **further tuning** to optimize for precision.  
🔹 **Neural Networks show promise**, but their **lack of explainability limits adoption in medical decision-making**.  

---

## **🚀 How to Run the Project**
### **1️⃣ Clone the Repository**
```bash
git clone https://github.com/your-username/heart-attack-prediction.git
cd heart-attack-prediction
```

### **2️⃣ Run the Jupyter Notebook**
```bash
jupyter notebook
```
- Open `heart-disease-prediction.ipynb`
- Follow the steps for **data preprocessing, model training, and evaluation**.


---

## **🔮 Future Work**
🔹 **Improve Precision-Recall Balance** – Adjust decision thresholds to minimize **false positives**.  
🔹 **Test Deployment in Clinical Settings** – Evaluate real-world usability on **hospital datasets**.  
🔹 **Explore Hybrid Models** – Try **ensembling Logistic Regression with XGBoost/LightGBM** to leverage strengths of each.  

---

## **📜 License**
This project is licensed under the **MIT License** – feel free to use and modify for research purposes.

---

## **🙌 Acknowledgments**
Special thanks to **[Kamil Pytlak](https://www.kaggle.com/kamilpytlak)** for providing the dataset.

📌 **For questions or collaboration, feel free to connect!**  
📩 Email: john@johnpospisil.com  
 🐦 [Twitter](https://x.com/audiofreak7)  
 🔗 [LinkedIn Profile](https://www.linkedin.com/in/johnpospisil/)
