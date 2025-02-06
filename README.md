## **YouFraud**

Real-time fraud detection system. This application is supposed to act as a validator for transactions. Transaction data would be passed in real-time and transaction validity would be determined in the context of fraudulency. Fraud transactions would be prompted to be terminated. Primary target of this application are banking systems.

## **Usage**

- The web interface can be accessed [here](https://youfraud.streamlit.app/).

- From github:

  ```bash
  git clone https://github.com/MetalInMyVeins/YouFraud
  cd YouFraud/
  pip install --no-cache-dir -r app/requirements.py
  streamlit run app/main.py
  ```

- Using docker:

  ```bash
  docker pull metalinmyveins/you_fraud-0.1.0:v1
  docker run -p 8501:8501 you_fraud-0.1.0
  ```

- The model is also hosted [here](https://huggingface.co/MetalInMyVeins/YouFraud).

- The dataset can be downloaded from [here](https://drive.google.com/file/d/1zg76nNbZXqkWnwH9umqg41iC1LeqiW-i/view?usp=drive_link).

## **How Does It Work**

We trained a machine learning model with a transaction dataset containing 6 Million+ transaction information performing various statistical analytical methods on it. The model can predict fraud transactions with 98% accuracy. The pipeline is presented below.

## **Project checklist**

### **Problem Identification**

- [x] Choosing real world problem
- [x] Outline the project goal
- [x] Identify primary audience

### **Data Collection**

- [ ] Web scraping
- [ ] API integration
- [x] Public datasets
  - [x] Kaggle
  - [ ] UCI
  - [ ] data.gov
  - [ ] Bangladesh Bank

### **Exploratory Data Analysis**

- [x] Create data summary report
- [x] Develop dashboard for data visualization

### **Data Preparation**

- [x] Split data into training set and test set

Apply data preprocessing techniques on training data:

- [x] Data Cleaning
  - [x] Handling missing values
  - [x] Outlier detection and treatment
  - [x] Data normalization or standardization
  - [x] Encoding categorical variables
- [x] Feature Engineering
  - [x] Create new features
  - [x] Transform existing features
- [x] Feature Selection
  - [x] Identify relevant features
  - [x] Remove irrelevant features
- [x] Document data preprocessing steps

- [x] Build a pipeline for data preprocessing to use the same pipeline on test data

### **Model Development and Evaluation**

- [x] Create baseline model
- [x] Experiment with different ML algorithms
- [x] Evaluate models with appropriate metrics
- [x] Perform hyperparameter tuning and model optimization
- [x] Implement ensemble methods for improved performance
- [x] Document model development process

### **Build Application**

- [X] Develop user-friendly interface
  - [X] Use Flask, FastAPI, Streamlit, Django

### **Model Deployment**

- [X] Share trained model:
  - [X] Hugging Face
  - [ ] Tensorflow Hub
- [X] Deploy application:
  - [ ] Heroku
  - [ ] AWS
  - [ ] Azure
  - [ ] Google Cloud Platform
  - [X] Docker
  - [ ] Kubernetes

## **License**

See [License](LICENSE) for more information.
