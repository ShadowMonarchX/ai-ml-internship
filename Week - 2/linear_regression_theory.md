
# ✅ **Data Preprocessing Tasks for Your CSV (`indian_food(in).csv`)**

Below are the tasks you should perform step-by-step.

---

## **1. Fix or Remove Outliers**

### **Tasks**

* Identify numerical columns (example: cooking time, ingredients count, rating, calories if present).
* Plot/inspect values that are unusually high or low.
* Check if the extreme values are:

  * **Valid** (e.g., a dish actually takes 240 minutes to cook) → keep.
  * **Invalid** (e.g., cooking time = 0, or 9999 minutes) → fix or remove the row.

### **How to fix**

* Replace invalid values with:

  * The median of the column
  * Or remove the row if it makes no sense

---

## **2. Encode Categorical Data**

Your dataset likely includes columns like:

* Cuisine
* Diet (Veg/Non-Veg)
* Course (Snack, Dessert, Main Course)
* Region/State
* Name (not needed for encoding)

### **Tasks**

* Decide which encoding to apply:

  * **One-Hot Encoding** → for columns with few categories (e.g., Diet).
  * **Label Encoding** → for large category columns (e.g., Cuisine, Region).
* Drop irrelevant columns (e.g., description text) if they cannot be encoded.

---

## **3. Normalize / Scale Numerical Features**

You must scale numerical columns so models work better.

### **Tasks**

* Identify numerical columns:

  * Cooking time
  * Prep time
  * Total time
  * Rating
  * Number of ingredients
* Choose scaling method:

  * **StandardScaler (Z-score)** → values become mean=0, std=1
  * **MinMaxScaler (0–1 range)** → values scaled between 0 and 1
* Apply scaling **after** splitting the data.

---

## **4. Split the Dataset**

### **Tasks**

Split into:

* **70% Train**
* **15% Validation**
* **15% Test**

### **What each set is used for**

* **Train:** Train the ML model
* **Validation:** Tune hyperparameters, choose best model
* **Test:** Final accuracy check

### **Important Rule**

👉 **Never scale before splitting**
Fit scalers only on **train**, then apply to validation & test.

---

# ⭐ **Why This Matters**

* Data preprocessing contributes to nearly **70% of model accuracy**.
* Clean, well-processed data → stable, generalizable models.
* Models struggle with:

  * Outliers
  * Unscaled numerical data
  * Categorical values in raw text

By completing these tasks correctly, your dataset becomes ML-ready.

---