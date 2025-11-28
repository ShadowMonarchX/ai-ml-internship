# **📅 Week 1 – Understanding Data (AI/ML Internship)**

## Day 1 Understanding Data
**Dataset:** Structured data for ML — rows = samples, columns = features.

---

## **Key Concepts**

* **Features (X):** Input values (e.g., `prep_time`, `cook_time`)
* **Target (Y):** Output to predict (e.g., `churn`)
* **Feature Types:** Numerical, Categorical, Boolean, Text
* **Data Distribution:** Mean, Median, Min/Max, Outliers
* **Missing Values:** Empty or `NaN`
* **Real-World Problem:** Understand purpose & business use-case


## **Importance**

* Avoid wrong predictions
* Choose correct model
* Prevent overfitting / underfitting
* Improve accuracy



## **Practical Tasks (With Code)**

* Load dataset in **Pandas**
* Inspect rows, columns, types, missing values
* Summary stats (mean, median, min/max)
* Visualize prep/cook times and state vs sweets (**Matplotlib**)
* Detect outliers (IQR method)


## **File**

📄 **Data Cleaning & Preprocessing.ipynb.ipynb**
[View Notebook](https://github.com/ShadowMonarchX/ai-ml-internship/blob/main/Week%20-%201/Data%20Cleaning%20%26%20Preprocessing.ipynb)

---

## Day 2 — Data Cleaning & Preprocessing

### Theory  
Raw data usually contains issues. Preprocessing makes it ready for modeling.  

**Key steps:**  
- Handle missing values  
- Remove duplicates  
- Fix or remove outliers  
 


### Task for Day 2  
1. Handle missing values in your dataset.  
2. Remove duplicate records if any.  
3. Identify and fix or remove outliers (as appropriate).  

---

## Day 3 — Missing Values & Data Correction

### Theory
Real datasets contain NA, blank, or incorrect values. Before modeling, data must be complete and consistent.   
(Later steps: encode categorical data, normalize/scale, train–validation–test split.)

### Tasks
1. Detect all NA / blank / null values.  
2. Fill missing values (numerical → mean/median, categorical → mode).  
3. Correct wrong or inconsistent values.  
4. Verify dataset is fully clean (no NA, no blanks, correct data types).  
5. Visualize cleaned data (histogram, box plot, count plot).
