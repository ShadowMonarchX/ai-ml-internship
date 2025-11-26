
---

# **📅 Day 1 – Understanding Data (AI/ML Internship)**

## **🔰 Topic: Introduction to Data & Dataset Understanding**

Today’s focus was on understanding the **foundation of AI/ML** — the **dataset**.
Before building any model, knowing the data deeply is the most important step.

---

## **📌 1. What is a Dataset?**

A **dataset** is a structured collection of data used for:

* Training ML models
* Testing model performance
* Understanding patterns
* Solving real-world problems

A dataset usually has:

* **Rows** → each sample/data point
* **Columns** → features/variables

---

## **📌 2. Key Concepts in Data Understanding**

### **✔ Features (X)**

Input values used to make predictions.
Examples: `age`, `income`, `prep_time`, `cook_time`.

### **✔ Target (Y)**

The value we want to predict.
Examples: `churn`, `fraud`, `price`.

### **✔ Feature Types**

* **Numerical:** prep_time, cook_time
* **Categorical:** state, region
* **Boolean:** true/false
* **Text:** ingredients list

### **✔ Data Distribution**

Understanding how values are spread:

* Mean
* Median
* Min/Max
* Outliers

### **✔ Missing Values**

Data that is empty or `NaN`.
These must be cleaned before training.

### **✔ Real-World Problem Understanding**

Before ML, ask:

* What problem are we solving?
* Why does it matter?
* What is the business use-case?

---

## **📌 3. Why Understanding Data is Important?**

* Prevents **wrong predictions**
* Helps choose the **correct ML model**
* Avoids **overfitting / underfitting**
* Improves **accuracy and reliability**

---

## **📌 4. Today’s Practical Work (No Code — Only Tasks Done)**

Using **Pandas** and **Matplotlib**, the following tasks were completed:

### **✔ Task 1: Load Dataset**

Loaded the “Indian Sweets Dataset” into Pandas.

### **✔ Task 2: View Basic Information**

* Checked rows & columns
* Identified feature types
* Checked missing values in `prep_time` and `cook_time`

### **✔ Task 3: Summary Statistics**

* Generated mean, median, min, max
* Detected strange values (e.g., `-1`, large times)

### **✔ Task 4: Visualizations**

Used **matplotlib** to plot:

* Prep Time distribution
* Cook Time distribution
* State vs Number of Sweets (bar chart)

### **✔ Task 5: Outlier Detection (Only Task, No Code)**

Applied **IQR method** to detect:

* unusually small prep/cook times
* unusually large prep/cook times

Understood how outliers affect ML models.

---

## **📌 5. Concepts Learned Today**

* What datasets are
* Meaning of features & target
* Importance of cleaning data
* Why missing values matter
* Role of outliers
* Real-world dataset analysis
* Basics of Pandas inspection
* Intro to Matplotlib plotting

---

## **📌 6. File for Day 1 Work**

Notebook:
📄 **Day-1.ipynb**

Link:
👉 [https://github.com/ShadowMonarchX/ai-ml-internship/blob/main/Week%20-%201/Day-1.ipynb](https://github.com/ShadowMonarchX/ai-ml-internship/blob/main/Week%20-%201/Day-1.ipynb)

---