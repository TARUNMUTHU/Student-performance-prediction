## Student Performance Predictor

This project predicts whether a student will pass or fail based on academic and demographic data using a trained machine learning model. It includes a web interface built with Flask.

---

##  Features

- Predicts **Pass/Fail** based on student data
- Trained using **Random Forest Classifier**
- Web interface using **Flask** and **Bootstrap**
- Form-based input for user-friendly prediction

---

##  Model Details

- **Dataset**: UCI Student Performance Dataset (Portuguese)
- **Target**: `G3` grade converted to Pass (≥10) or Fail (<10)
- **Features Used**:
  - sex, age, address, family size, parental status
  - mother’s and father’s education
  - study time, failures, absences, G1, G2

---

## Technologies Used

- Python
- Pandas, Scikit-learn, Joblib
- Flask
- Bootstrap (for frontend styling)

---

##  How to Run Locally

1. Clone the repository:
   ```bash
   git clone https://github.com/TARUNMUTHU/Student-performance-prediction.git
   cd Student-performance-prediction
