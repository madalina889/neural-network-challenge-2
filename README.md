# neural-network-challenge-2
# Employee Attrition Prediction

This project uses a deep learning model to predict employee attrition (whether an employee is likely to leave the company) based on a variety of HR-related features. It is built using TensorFlow/Keras and includes preprocessing, feature engineering, and evaluation steps.

---

## 📊 Dataset

The dataset includes features such as:
- Age
- Department
- Education
- Distance from Home
- Job Satisfaction
- OverTime status
- Work-Life Balance
- Total Working Years
- And more...

Target column:
- **Attrition** (Yes/No)

---

## 🧪 Preprocessing Steps

- Selected 10+ relevant columns for `X` features.
- Created `y_df` using `Attrition` and `Department`.
- Encoded categorical features:
  - `OverTime`, `Department`, and `Attrition` were one-hot or label encoded.
- Standardized numerical features using `StandardScaler`.
- Split the dataset into training and testing sets (80/20 split).

---

## 🧠 Model Architecture

A multi-input deep learning model was built using the **Keras Functional API**, consisting of:

### Main Input (X features)
- Input Layer
- Dense Layer (64 units, ReLU)
- Dense Layer (32 units, ReLU)

### Department Branch
- Input Layer (One-hot encoded)
- Dense Layer (8 units, ReLU)
- Output to merge (4 units, ReLU)

### Attrition Branch
- Input Layer (One-hot or binary)
- Dense Layer (8 units, ReLU)
- Output to merge (4 units, ReLU)

### Final Layers
- Concatenation of the three branches
- Dense Layer (16 units, ReLU)
- Output Layer (1 unit, **sigmoid** for binary classification)

---

## ⚙️ Model Compilation and Training

```python
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(
    x={
        'main_input': X_train_scaled,
        'department_input': X_train_dept,
        'attrition_input': y_train_encoded
    },
    y=y_train_binary,
    validation_split=0.2,
    epochs=20,
    batch_size=32
)
