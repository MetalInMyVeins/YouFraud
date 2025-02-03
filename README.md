## YouFraud

Real-time fraud detection system. This application is supposed to act as a validator for transactions. Transaction data would be passed in real-time and transaction validity would be determined in the context of fraudulency. Fraud transactions would be prompted to be terminated. Primary target of this application are banking systems.

## Project checklist

### Problem Identification

- [x] Choosing real world problem
- [x] Outline the project goal
- [x] Identify primary audience

### Data Collection

- [ ] Web scraping
- [ ] API integration
- [x] Public datasets
  - [x] Kaggle
  - [ ] UCI
  - [ ] data.gov
  - [ ] Bangladesh Bank

### Exploratory Data Analysis

- [x] Create data summary report
- [x] Develop dashboard for data visualization

### Data Preparation

- [ ] Split data into training set and test set

Apply data preprocessing techniques on training data:

- [ ] Data Cleaning
  - [ ] Handling missing values
  - [ ] Outlier detection and treatment
  - [ ] Data normalization or standardization
  - [ ] Encoding categorical variables
- [ ] Feature Engineering
  - [ ] Create new features
  - [ ] Transform existing features
- [ ] Feature Selection
  - [ ] Identify relevant features
  - [ ] Remove irrelevant features
- [ ] Document data preprocessing steps

- [ ] Build a pipeline for data preprocessing to use the same pipeline on test data

### Model Development and Evaluation

- [ ] Create baseline model
- [ ] Experiment with different ML algorithms
- [ ] Evaluate models with appropriate metrics
- [ ] Perform hyperparameter tuning and model optimization
- [ ] Implement ensemble methods for improved performance
- [ ] Document model development process

### Build Application

- [ ] Develop user-friendly interface
  - [ ] Use Flask, FastAPI, Streamlit, Django

### Model Deployment

- [ ] Share trained model:
  - [ ] Hugging Face
  - [ ] Tensorflow Hub
- [ ] Deploy application:
  - [ ] Heroku
  - [ ] AWS
  - [ ] Azure
  - [ ] Google Cloud Platform
  - [ ] Docker
  - [ ] Kubernetes

