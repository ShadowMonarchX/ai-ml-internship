# 🧠 Machine Learning & AI Glossary: A Foundational Reference

## 🌟 Overview and Structure

This glossary defines critical concepts, algorithms, metrics, and ethical considerations in the field of Artificial Intelligence (AI).

### 🏷️ Key Categories & Tags
To help categorize and understand the context of each term, we use the following tags:

| Tag | Description |
| :--- | :--- |
| **#fundamentals** | Core concepts applicable across most ML domains. |
| **#Metric** | Terms used to evaluate model performance (Loss, Accuracy, AUC, etc.). |
| **#generativeAI** | Concepts specific to Large Language Models (LLMs) and Generative systems. |
| **#responsible** | Terms related to ML ethics, fairness, and bias. |
| **#df** | Techniques related to Decision Forests (Trees, Random Forests, Boosting). |
| **#clustering** | Unsupervised learning techniques for grouping data. |
| **#GoogleCloud** | Terms related to ML infrastructure and hardware. |

---

## 🅰️ A: Foundational Concepts & Advanced Techniques

### 1. **Ablation**
* **Definition:** A technique to determine the **importance of a feature or component** by temporarily removing it from a trained model and observing the resulting performance change.
* **Step-by-Step Evaluation:**
    1.  Train the model ($M_{full}$) with all features/components. Record performance ($P_{full}$).
    2.  Temporarily remove component $C_i$ (ablation).
    3.  Retrain the model ($M_{ablated}$) without $C_i$. Record performance ($P_{ablated}$).
    4.  **Conclusion:** If $P_{full}$ is significantly better than $P_{ablated}$, $C_i$ was important.

### 2. **Accuracy** **#fundamentals** **#Metric**
* **Definition:** The fraction of **correct classification predictions** out of the total number of predictions.
* **Formula (Binary Classification):**
    $$
    \text{Accuracy} = \frac{\text{True Positives (TP)} + \text{True Negatives (TN)}}{\text{TP} + \text{TN} + \text{False Positives (FP)} + \text{False Negatives (FN)}}
    $$

### 3. **Activation Function** **#fundamentals**
* **Definition:** A function applied to the output of a neuron that introduces **nonlinearity** into the neural network, enabling it to learn complex, non-linear relationships.
* **Step-by-Step Function:**
    1.  **Linear Sum:** Inputs are multiplied by **weights** and summed with the **bias**.
    2.  **Activation:** The sum is passed through the activation function.
    3.  **Output:** The result is passed to the next layer.
* **Popular Examples:** **ReLU** (Rectified Linear Unit), **Sigmoid**.


### 4. **Agent & Action** **#generativeAI**
* **Agent (Generative AI):** Software capable of **reasoning, planning, and executing actions** on behalf of the user (e.g., using external tools).
* **Agent (Reinforcement Learning):** The entity that uses a **policy** to choose an **action** to transition between **states** in an environment to maximize expected reward.

### 5. **Artificial Intelligence (AI)** **#fundamentals**
* **Definition:** A non-human program or model that can solve **sophisticated tasks**, such as translating text or identifying images. Machine Learning is a sub-field of AI.

### 6. **Attention**
* **Definition:** A mechanism in neural networks (central to **Transformers**) that dynamically **weighs the importance** of different parts of the input (e.g., words in a sentence) when processing a specific part. It helps compress necessary information.

### 7. **AUC (Area under the ROC curve)** **#fundamentals** **#Metric**
* **Definition:** A scalar value (0.0 to 1.0) representing a **binary classification** model's ability to **separate positive classes from negative classes** across all possible **classification thresholds**.
    * **1.0** = Perfect separation.
    * **0.5** = Random results (no separation).
* **ROC Curve:** AUC is the area under the **Receiver Operating Characteristic (ROC)** curve, which plots the **True Positive Rate (Recall)** vs. the **False Positive Rate** at different thresholds.


### 8. **Autoencoder**
* **Definition:** A system used for learning efficient data encodings (representations), consisting of two parts trained end-to-end:
    * **Encoder:** Maps input to a lower-dimensional (lossy) intermediate format.
    * **Decoder:** Maps the intermediate format back to a version of the original input.
* **Purpose:** Data compression, noise reduction, and representation learning.


---

## 🅱️ B: Model Training & Bias

### 1. **Backpropagation** **#fundamentals**
* **Definition:** The core algorithm that implements **Gradient Descent** in neural networks. It calculates the **gradient** (error derivative) of the **loss function** with respect to every **weight** in the network.
* **Two-Pass Cycle:**
    1.  **Forward Pass:** Calculate predictions and the total **loss** for the current batch.
    2.  **Backward Pass (Backpropagation):** Use the **chain rule** of calculus to distribute the total loss back through the layers, determining how much each weight contributed to the error.
    3.  **Weight Adjustment:** Update all weights using the **learning rate** multiplier.


### 2. **Bagging (Bootstrap Aggregating)** **#df**
* **Definition:** An **ensemble** technique where multiple models (e.g., **Decision Trees** in a **Random Forest**) are trained on different, random subsets of the training data, sampled **with replacement** (bootstrap).
* **Purpose:** Reduces **variance** and prevents **overfitting**.

### 3. **Batch & Batch Size** **#fundamentals**
* **Batch:** The subset of examples used in a **single training iteration**.
* **Batch Size:** The number of examples in the batch.
* **Strategies:**
    * **Stochastic Gradient Descent (SGD):** Batch Size = 1.
    * **Mini-Batch:** Batch Size = 10 to 1,000 (Most efficient).
    * **Full Batch:** Batch Size = Entire training set (Least efficient).

### 4. **BERT (Bidirectional Encoder Representations from Transformers)**
* **Architecture:** Uses the **Encoder** stack of the **Transformer** architecture.
* **Key Feature:** **Bidirectional** processing, meaning it evaluates the context of a word based on both the text **preceding** and **following** it. This is achieved using **masking** for unsupervised training.

### 5. **Bias (Math/Bias Term)** **#fundamentals**
* **Definition:** An intercept ($b$ or $w_0$) in a linear model equation ($\hat{y} = w_1x_1 + b$). It represents the output when all feature inputs ($x_i$) are zero.
* **Role:** Allows the model to shift the prediction line (or hyperplane) away from the origin.

### 6. **Bias (Ethics/Fairness)** **#responsible**
* **Definition:** Systematic error or favoritism that can lead to unfair or prejudicial outcomes.
* **Categories:**
    * **Social Bias:** Stereotyping or prejudice (e.g., **Automation Bias**, **Implicit Bias**).
    * **Systemic Bias:** Error in data or procedure (e.g., **Coverage Bias**, **Sampling Bias**).

### 7. **Binary Classification** **#fundamentals**
* **Definition:** A classification task that predicts one of **two mutually exclusive classes** (e.g., Positive vs. Negative, Spam vs. Not Spam).
* **Key Tool:** **Classification Threshold** is used to convert the model's probability score into a final class prediction.

### 8. **Bucketing (Binning)** **#fundamentals**
* **Definition:** Converting a single **continuous feature** (e.g., temperature) into multiple **discrete, binary features** (buckets) based on defined value ranges.
* **Benefit:** The model treats all values within the same bucket identically, which can help capture non-linearities in data.
* **Example:** Converting temperature into 'Cold' ($\le 10^\circ$C), 'Temperate' ($11^\circ-24^\circ$C), and 'Warm' ($\ge 25^\circ$C).

---

## 🇨 C: Evaluation & Data Preparation

### 1. **Candidate Generation**
* **Definition:** The **initial, computationally inexpensive** phase of a recommendation system that quickly filters a massive catalog (e.g., 100,000 items) down to a manageable shortlist of **suitable items** (e.g., 500 items) for a user.

### 2. **Categorical Data** **#fundamentals**
* **Definition:** Features that represent discrete, named categories (e.g., 'Color', 'City', 'Animal Type').
* **Processing:** Must typically be converted to a numerical format using techniques like **One-Hot Encoding** or **Embedding**.

### 3. **Classification Threshold** **#fundamentals**
* **Definition:** The cutoff point (a probability value between 0 and 1) used in **binary classification** to determine the final class output.
    * **Prediction > Threshold** $\rightarrow$ Positive Class.
    * **Prediction $\le$ Threshold** $\rightarrow$ Negative Class.

### 4. **Clustering** **#clustering**
* **Definition:** An **unsupervised learning** technique that automatically groups similar data examples into sets called **clusters**. Similarity is defined based on the feature values.
* **Algorithm Example:** **Hierarchical Clustering** (Agglomerative).

### 5. **Confusion Matrix** **#Metric**
* **Definition:** A $2 \times 2$ table summarizing the outcomes of a **binary classification** model, explicitly showing all four possible results:
    | | Predicted Positive | Predicted Negative |
    | :--- | :--- | :--- |
    | **Actual Positive** | **True Positive (TP)** | **False Negative (FN)** |
    | **Actual Negative** | **False Positive (FP)** | **True Negative (TN)** |


### 6. **Convergence**
* **Definition:** The state during model training when the **loss** stabilizes or changes negligibly. It indicates that the model has likely reached the minimum point (or a local minimum) on the loss landscape.

### 7. **Cross-Entropy (Log Loss)**
* **Definition:** A common **loss function** used in **classification** tasks. It measures the difference between the true probability distribution (the **label**) and the predicted probability distribution output by the model.

---

## 🇩 D: Depth and Dimensionality

### 1. **Data Augmentation**
* **Definition:** Techniques used to synthetically **increase the size and diversity** of a training dataset by applying minor transformations to existing examples (e.g., flipping or rotating images, paraphrasing text).
* **Goal:** Improves model generalization and reduces **overfitting**.

### 2. **Decision Tree (DT)** **#df**
* **Definition:** A flow-chart-like model that uses a sequence of **axis-aligned** (single-feature) or **oblique** (multi-feature) conditions to recursively partition the feature space until a prediction is reached at a leaf node.

### 3. **Decoder** **#generativeAI**
* **Definition:** The component in a sequence-to-sequence or **Transformer** model that takes the processed internal representation (the **encoding**) and sequentially generates the final output (e.g., translating a sentence, generating the next tokens).

### 4. **Deep Model**
* **Definition:** A neural network that has **multiple hidden layers**. The term "deep learning" refers to models with this structural depth.

### 5. **Dropout**
* **Definition:** A powerful **regularization** technique for neural networks where a fixed percentage of neurons in a layer are **randomly ignored** (set to zero) during each training iteration.
* **Mechanism:** Prevents neurons from co-adapting too much, leading to more robust feature representations and reduced **overfitting**.

---

## 🇪 E: Ensemble and Evaluation

### 1. **Early Stopping**
* **Definition:** A **regularization** technique where model training is halted when performance on a separate **validation set** starts to degrade (after an initial improvement), even if the loss on the training set is still decreasing.
* **Purpose:** Prevents the model from progressing into the **overfitting** phase.

### 2. **Embedding** **#fundamentals**
* **Definition:** A relatively **low-dimensional, learned numerical vector** that represents high-dimensional categorical or complex data (like words, users, or items).
* **Semantic Relationship:** Embeddings capture the semantic meaning such that items with similar meanings or properties have vectors that are mathematically close together.

### 3. **Encoder** **#generativeAI**
* **Definition:** The component in a sequence-to-sequence or **Transformer** model that processes the input sequence and transforms it into a rich, internal, context-aware representation or **embedding**.

### 4. **Ensemble** **#df**
* **Definition:** An approach that combines the predictions of multiple individual models (**base estimators**) to produce a single, typically more accurate and robust prediction.
* **Strategies:** **Bagging** (like Random Forests) and **Boosting** (like Gradient Boosted Trees).

### 5. **Epoch** **#fundamentals**
* **Definition:** One complete pass through the **entire training dataset**. An epoch consists of $\lceil \text{Total Examples} / \text{Batch Size} \rceil$ iterations.

---

## 🇫 F: Features and Foundation

### 1. **False Negative (FN) & False Positive (FP)** **#Metric**
* **False Negative (FN):** Model predicts Negative, but the label is Positive (The model missed the true event).
* **False Positive (FP):** Model predicts Positive, but the label is Negative (The model raised a false alarm).

### 2. **Feature** **#fundamentals**
* **Definition:** An input variable or measurable characteristic used by the model to make a prediction.
* **Synonym:** **Attribute**.

### 3. **Feature Cross** **#fundamentals**
* **Definition:** A synthetic feature created by multiplying, crossing, or combining two or more individual features.
* **Purpose:** Allows a model to learn **interaction effects** and non-linear relationships specific to the unique combination of feature values (e.g., $\text{Latitude} \times \text{Longitude}$).

### 4. **Few-Shot Learning** **#generativeAI**
* **Definition:** The ability of a **LLM** to learn a new task and generalize well after being provided with only **a small number of examples** (typically 3-10) directly within the input **prompt**.

### 5. **Fine-Tuning** **#generativeAI**
* **Definition:** The process of taking a **Pre-trained Model** (or **Foundation Model**) and continuing its training on a **smaller, task-specific dataset** to adapt its weights for a specialized task (e.g., medical text summarization).

### 6. **Foundation Model** **#generativeAI**
* **Definition:** A very large ML model (often a **LLM**) trained on a massive, broad dataset, designed to be a starting point that can be adapted (via **fine-tuning** or **prompting**) for a wide range of downstream applications.

---

## 🇬 G: Generation and Gradient

### 1. **Generative AI** **#generativeAI**
* **Definition:** A category of AI that focuses on creating **novel, synthetic content** (e.g., text, images, code, audio) that is original but statistically similar to the training data.

### 2. **Gradient** **#fundamentals**
* **Definition:** In the context of the loss function, the gradient is a vector that points in the direction of the **steepest increase** in loss.
* **Role in Training:** **Gradient Descent** moves in the opposite direction (negative gradient) to minimize loss.

### 3. **Gradient Descent** **#fundamentals**
* **Definition:** The optimization algorithm used to train models by iteratively adjusting **weights** and **bias** in the direction that **minimizes the loss function**.
* **Key Parameters:** **Gradient** (direction) and **Learning Rate** (step size).

---

## 🇭 H: Hidden and Hyperparameters

### 1. **Hierarchical Clustering (Agglomerative Clustering)** **#clustering**
* **Definition:** A **clustering** method that builds a hierarchy of clusters by starting with each data point as its own cluster and then iteratively **merging** the closest pairs of clusters until all points belong to a single, large cluster.

### 2. **Hidden Layer** **#fundamentals**
* **Definition:** Any layer in a neural network positioned between the input layer and the output layer. These layers are responsible for learning increasingly complex and abstract representations of the input data.

### 3. **Hyperparameter** **#fundamentals**
* **Definition:** A configuration variable that is set **manually before training** and controls the training process, rather than being learned by the model itself.
* **Examples:** **Learning Rate**, **Batch Size**, number of layers, number of trees in a **Decision Forest**.

---

## 🇮 I: Inference and Interpretability

### 1. **Inference**
* **Definition:** The process of a **trained model** making a prediction on a new, **unlabeled example**.
* **Types:** **Batch Inference** (many examples at once) vs. dynamic (real-time) inference.

### 2. **Interpretability**
* **Definition:** The degree to which a human can understand the **mechanisms and reasoning** behind a model's prediction.
* **Contrast:** Low interpretability models are often called **Black Box Models**.

---

## 🇱 L: Language and Loss

### 1. **Label** **#fundamentals**
* **Definition:** The "answer" or true value that a model is trained to predict in **supervised learning**.

### 2. **Large Language Model (LLM)** **#generativeAI**
* **Definition:** A **Foundation Model** characterized by billions or trillions of **parameters**, trained on massive text data, and capable of generating, summarizing, and reasoning over human language. Most LLMs are **auto-regressive**.

### 3. **Learning Rate** **#fundamentals**
* **Definition:** A **hyperparameter** that determines the size of the adjustment step taken during **Gradient Descent**.
    * Too high $\rightarrow$ Divergence/overshooting the minimum.
    * Too low $\rightarrow$ Very slow convergence.

### 4. **Loss** **#fundamentals**
* **Definition:** A numerical value that quantifies the **penalty for a bad prediction**. The primary objective of training is to minimize this value.
* **Examples:** **Mean Squared Error** (Regression), **Cross-Entropy** (Classification).

### 5. **Logistic Regression** **#fundamentals**
* **Definition:** A linear model primarily used for **binary classification**. It applies the **Sigmoid** activation function to the output of a linear model to squash the result into a probability between 0 and 1.

---

## Ⓜ️ M: Metrics and Methods

### 1. **Machine Learning (ML)** **#fundamentals**
* **Definition:** A sub-field of AI where computer systems learn from data to make predictions or decisions without explicit programming.

### 2. **Masking**
* **Definition:** A training technique, often used in **BERT**, where specific tokens in the input are hidden, and the model is trained to predict the original identity of the masked tokens, thereby learning context bidirectionally.

### 3. **Mean Squared Error (MSE)** **#Metric**
* **Definition:** The average of the squares of the errors (the difference between the actual value and the predicted value). It is the most common **loss function** for **regression** tasks.
    $$
    \text{MSE} = \frac{1}{N}\sum_{i=1}^{N}(y_i - \hat{y}_i)^2
    $$

### 4. **Multi-Class Classification** **#fundamentals**
* **Definition:** A classification task where the model predicts one of **more than two** mutually exclusive classes.
* **Common Output Layer:** Uses the **Softmax** function.

---

## 🅾️ O: Optimization and Outcomes

### 1. **One-Hot Encoding** **#fundamentals**
* **Definition:** The standard technique for converting **categorical data** into a numerical format. Each category is represented by a binary vector where only the corresponding element is 1 (hot) and all others are 0.
* **Result:** Creates **Sparse Features**.

### 2. **Overfitting** **#fundamentals**
* **Definition:** The model learns the **training data too well**, including its noise and idiosyncrasies, resulting in **excellent performance on training data** but **poor generalization** to new, unseen data (the test set).
* **Mitigation:** **Regularization**, **Dropout**, **Early Stopping**.

---

## 🅿️ P: Performance and Probability

### 1. **Precision** **#Metric**
* **Definition:** Out of all predictions the model made as **Positive**, how many were **actually correct** (True Positives)?
    $$
    \text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}
    $$
* **Focus:** Minimizing **False Positives** (False Alarms).

### 2. **Pre-Trained Model** **#generativeAI**
* **Definition:** A model that has completed its initial, expensive training phase on a massive, general dataset and whose **weights are saved**. It is ready for immediate deployment or specialized adaptation via **Fine-Tuning**.

### 3. **Prompt** **#generativeAI**
* **Definition:** The input (text, multimodal, etc.) provided to a **Generative AI** model to instruct it to produce a specific output.

---

## 🇷 R: Reward and Reduction

### 1. **Recall** **#Metric**
* **Definition:** Out of all the **actual Positive** examples, how many did the model **correctly identify**?
    $$
    \text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}
    $$
* **Focus:** Minimizing **False Negatives** (Missed Opportunities).

### 2. **Regularization** **#fundamentals**
* **Definition:** Any technique used to explicitly **penalize model complexity** during training to reduce **overfitting**.
* **Examples:** L1/L2 penalties on weights, **Dropout**, **Early Stopping**.

### 3. **Reinforcement Learning (RL)**
* **Definition:** A machine learning paradigm where an **Agent** learns an optimal **Policy** by taking **Actions** in an **Environment** to maximize a cumulative **Reward**.

### 4. **ReLU (Rectified Linear Unit)**
* **Definition:** A popular **Activation Function**. $\text{ReLU}(x) = \max(0, x)$. It sets all negative inputs to zero and leaves positive inputs unchanged.


### 5. **ROC Curve**
* **Definition:** See **AUC**.

---

## 🇸 S: Structures and Supervision

### 1. **Self-Attention**
* **Definition:** A key component of the **Transformer** architecture. It calculates the relevance of every token in a sequence to every other token, allowing the model to weigh different parts of the context simultaneously to form an informed representation.

### 2. **Sigmoid**
* **Definition:** An **Activation Function** often used in **Logistic Regression** that squashes any input value into the probability range (0, 1).


[Image of the Sigmoid function plot]


### 3. **Softmax**
* **Definition:** A function used in the output layer of **Multi-Class Classification** models that converts a vector of raw scores (logits) into a probability distribution, ensuring all resulting probabilities sum to 1.

### 4. **Supervised Learning**
* **Definition:** The category of ML where the model is trained on data consisting of input **Features** paired with desired output **Labels**.
* **Examples:** **Classification**, **Regression**.

---

## 🇹 T: Training and Transformers

### 1. **Token** **#generativeAI**
* **Definition:** The basic unit of data (often a word, part of a word, or punctuation) that **LLMs** process and generate.

### 2. **Transformer** **#generativeAI**
* **Definition:** The model architecture, based entirely on **Self-Attention** and **Multi-Head Self-Attention**, that underpins almost all modern **LLMs** (BERT, GPT, Gemini). It replaced recurrent neural networks (RNNs) for sequence processing.

### 3. **True Negative (TN) & True Positive (TP)** **#Metric**
* **True Negative (TN):** Model correctly predicted Negative.
* **True Positive (TP):** Model correctly predicted Positive.

---

## 🇺 U: Uncertainty and Unsupervised

### 1. **Unsupervised Learning**
* **Definition:** The category of ML where the model is trained on **unlabeled data** to find inherent patterns, structure, or groupings within the data.
* **Examples:** **Clustering**, **Autoencoders**.

---

## 🇻 V: Validation and Vanishing

### 1. **Validation Set**
* **Definition:** A subset of data, separate from the training and test sets, used to monitor model performance **during training** to tune **hyperparameters** and determine when to use **Early Stopping**.

### 2. **Vanishing Gradient Problem**
* **Definition:** A common issue in deep networks where the **Gradient** becomes infinitesimally small as it propagates back through the initial layers during **Backpropagation**, causing the weights in those layers to update too slowly, hindering learning.

---

## 🇼 W: Weight

### 1. **Weight** **#fundamentals**
* **Definition:** The primary **parameter** learned by a model. It determines the **strength** of the connection between an input feature/neuron and the output/next neuron. Weights are adjusted by **Gradient Descent**.

---

## 🇿 Z: Zero-Shot

### 1. **Zero-Shot Learning** **#generativeAI**
* **Definition:** The ability of a **LLM** to perform a task it was **never explicitly trained for** by only relying on the instruction provided in the **prompt**, with **zero examples** given in the input.

***

Would you like to explore the core formula for **Precision and Recall** side-by-side using the **Confusion Matrix** terms?