# 🚀 Fake Review Detector: AI-Powered Review Classification  

## 📌 Overview  
Fake reviews mislead consumers and harm businesses. This **Fake Review Detector** uses **Machine Learning & NLP** to classify reviews as **genuine (OR) or fake (FR)**.  

By leveraging **TF-IDF vectorization, CountVectorizer, and ensemble ML models**, the system identifies deceptive patterns in review texts.  

---

## 🔥 Features  
✅ **Machine Learning-Based Classification**  
- Implements **Naïve Bayes, Random Forest, Decision Tree, KNN, SVM, and Logistic Regression**.  
- Uses **Bag-of-Words (BoW) and TF-IDF** for feature extraction.  

📊 **Text Processing & Feature Engineering**  
- Removes **stopwords, punctuation, and special characters**.  
- Extracts **top influential words** affecting classification.  

📈 **Performance Evaluation**  
- Computes **Accuracy, Precision, Recall, and F1-score**.  
- Displays **Confusion Matrices & Model Comparisons**.  

🛠 **Dataset Handling**  
- Loads **Preprocessed Fake Reviews Detection Dataset.csv**.  
- Analyzes **text length distribution** for data insights.  

---

## 📊 Results
The classification models were evaluated based on Accuracy, Precision, Recall, and F1-score. Below is a summary of their performance:

🔹 **Model Performance**

![image](https://github.com/user-attachments/assets/147b3b9f-65dc-450d-8db2-11248fe3d3ad)


🔹 **Key Takeaways**

- Best Performing Model: SVM (88.23%) achieved the highest accuracy and F1-score.

- Naïve Bayes (85.15%) and Logistic Regression (86.48%) also showed strong performance.

- Decision Tree (73.75%) and KNN (70.52%) underperformed due to overfitting and sensitivity to feature space.

🔹 **Most Influential Words**

Words most affecting classification:

small, littl, bought, much, problem, nt, good, even, great, love

<!--![image](https://github.com/user-attachments/assets/e8370b61-244e-4803-80eb-193362d9d669)
![image](https://github.com/user-attachments/assets/442d7138-6315-4a09-bf67-9cf4383df24e)
![image](https://github.com/user-attachments/assets/6f716199-2dc0-4cda-85c1-edf0d5643094)
![image](https://github.com/user-attachments/assets/5419c433-0dc3-4a48-9f71-e8ddd54ef4c9)
![image](https://github.com/user-attachments/assets/94aa15fd-1ced-4d0a-ac87-85625533f102)
![image](https://github.com/user-attachments/assets/36820110-3b65-4a83-8f05-cfcd2c5ccb50)
![image](https://github.com/user-attachments/assets/1951bd0e-cd65-4be3-a1ff-852c1b2cc732)-->



## Access the project code on **Google Colab**:  
🔗 [Colab Link]([https://colab.research.google.com/drive/1EHJ3MnVA3v58g9QradRSbT8b5mCq4lwp?usp=sharing](https://colab.research.google.com/drive/1nqTG-gdre1JYUR7yMtT6AipsvHpzCkn9?usp=sharing))

## ⚡ Installation  
Clone the repository:  
```bash
git clone https://github.com/yourusername/Fake-Review-Detector.git
cd Fake-Review-Detector

