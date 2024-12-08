# Stock Movement Prediction

This repository contains the code and resources for predicting stock movements based on sentiment analysis of Reddit posts. The project involves data scraping, preprocessing, feature engineering, and training a logistic regression model to predict stock trends.

---

## Table of Contents

1. [Setup](#setup)  
2. [Data Scraping](#data-scraping)  
3. [Data Preprocessing](#data-preprocessing)  
4. [Sentiment Analysis](#sentiment-analysis)  
5. [Model Analysis](#model-analysis)  
6. [Model Training](#model-training)  
7. [Model Testing](#model-testing)  
8. [Running the Code](#running-the-code)  
9. [Acknowledgements](#acknowledgements)  

---

## Setup

### Prerequisites
To run this project, you need:
- Python 3.x
- pip
- Reddit API credentials (client ID, client secret, user agent)
- Required Python libraries:
  - `pandas`
  - `praw`
  - `textblob`
  - `scikit-learn`
  - `joblib`
  - `imblearn`

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/stock-movement-prediction.git
   cd stock-movement-prediction
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Data Scraping

The `stockscraper.py` script uses the PRAW library to scrape data from Reddit subreddits.

### Steps:
1. Replace the Reddit API credentials in the script:
   ```python
   client_id = "your_client_id"
   client_secret = "your_client_secret"
   user_agent = "your_user_agent"
   ```
2. Run the script to scrape posts:
   ```bash
   python stockscraper.py
   ```
3. **Output**: A CSV file named `reddit_stock_data.csv` containing raw post data from subreddits like WallStreetBets, StockMarket, and Investing.

---

## Data Preprocessing

The `cleaned_data.py` script processes raw Reddit data by:
- Combining the title and selftext columns.
- Cleaning text (removing URLs, special characters, and extra spaces).
- Keeping only relevant columns like `content`, `score`, `num_comments`, and `created_utc`.

### Steps:
1. Ensure `reddit_stock_data.csv` exists in the directory.
2. Run the script:
   ```bash
   python cleaned_data.py
   ```
3. **Output**: A CSV file named `cleaned_reddit_stock_data.csv`.

---

## Sentiment Analysis

The `sentiment_analysis.py` script extracts sentiment and other features necessary for stock prediction.  

### Key Tasks:
1. **Sentiment Polarity**: Calculates sentiment polarity using TextBlob. 
2. **Stock Mentions**: Counts how often the term "stock" is mentioned in the post.  
3. **Time Features**: Extracts the post creation hour from the timestamp.  
4. **Engagement Metrics**: Includes the number of comments and post score.  

### Steps:
1. Ensure `cleaned_reddit_stock_data.csv` exists in the directory.
2. Open `sentiment_analysis.py` and verify its configurations if required.
3. Run the script:
   ```bash
   python sentiment_analysis.py
   ```


## Model Analysis

The `model_analysis.py` script processes the data for machine learning.  

### Key Tasks:
1. **Feature and Target Splitting**: Separates the independent variables (`X`) from the target variable (`y`).  
2. **Data Splitting**: Splits the dataset into training and testing sets (e.g., 80% training, 20% testing).  

### Steps:
1. Run the script:
   ```bash
   python model_analysis.py
   ```
2. **Outputs**:
   - `X_train.csv`, `X_test.csv`: Features for training and testing.  
   - `y_train.csv`, `y_test.csv`: Labels for training and testing.  

---

## Model Training

The `train_model.py` script trains a logistic regression model to predict stock movements.  

### Key Tasks:
1. **Model Training**: Fits a logistic regression model to the training data.  
2. **Model Evaluation**: Calculates accuracy and generates a classification report.  
3. **Model Saving**: Saves the trained model for future use.  

### Steps:
1. Ensure the preprocessed files (`X_train.csv`, `X_test.csv`, `y_train.csv`, `y_test.csv`) exist in the directory.
2. Run the script:
   ```bash
   python train_model.py
   ```
3. **Outputs**:
   - Model evaluation metrics: Accuracy and classification report.  
   - Trained model saved as `stock_movement_model.pkl`.  

---

## Model Testing

The `test_model.py` script performs additional testing of the trained model.  

### Key Tasks:
1. **Handling Class Imbalance**: Uses SMOTE (Synthetic Minority Oversampling Technique) to address class imbalance.  
2. **Cross-validation**: Re-splits data into training and validation sets for robust testing.  

### Steps:
1. Ensure the preprocessed files and model (`X_train.csv`, `X_test.csv`, `y_train.csv`, `y_test.csv`, `stock_movement_model.pkl`) exist in the directory.
2. Run the script:
   ```bash
   python test_model.py
   ```
3. **Outputs**:
   - Evaluation metrics after additional testing.  
   - Updated model saved as `stock_movement_model.pkl`.  

---

## Running the Code

To execute the pipeline, follow these steps in order:

1. **Data Scraping**:
   ```bash
   python stockscraper.py
   ```
2. **Data Preprocessing**:
   ```bash
   python cleaned_data.py
   ```
3. **Sentiment Analysis**:
   ```bash
   python sentiment_analysis.py
   ```
4. **Data Splitting (Model Analysis)**:
   ```bash
   python model_analysis.py
   ```
5. **Model Training**:
   ```bash
   python train_model.py
   ```
6. **Model Testing**:
   ```bash
   python test_model.py
   ```

---

## Acknowledgements

- **PRAW** for Reddit API integration.  
- **TextBlob** for sentiment analysis.  
- **Pandas** for data preprocessing and manipulation.  
- **scikit-learn** for machine learning algorithms.  
- **imblearn** for addressing class imbalance.  
