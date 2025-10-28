# **SPAM SMS CLASSIFICATION**

It is a ML project made for **classification of messages(SPAM/HAM).**
Also the model can be directly accesed using a web app build using **Streamlit**.
For detailed model evaluation and analysis see: **[Analysis](https://github.com/Mr-Atanu-Roy/Spam-SMS-Classification/blob/master/analysis.md)**

**Demos:**
<img width="691" height="330" alt="Screenshot 2025-10-28 163527" src="https://github.com/user-attachments/assets/71eab422-307c-4f35-bce8-ef1c0437a1c9" />
<img width="609" height="397" alt="Screenshot 2025-10-28 163535" src="https://github.com/user-attachments/assets/aa4cc4d3-faef-4086-843f-4506bdd4ab75" />



**Tech Stack:**

-   **Python:** The primary programming language used for the project.
-   **Pandas:** Used for data manipulation and analysis.
-   **NumPy:** Used for numerical operations.
-   **Matplotlib & Seaborn:** Used for data visualization.
-   **NLTK:** Used for natural language processing tasks like tokenization, stop word removal, and stemming.
-   **Scikit-learn:** Used for machine learning model building, including text vectorization (CountVectorizer, TfidfVectorizer), model training (Naive Bayes classifiers), and evaluation metrics.
-   **WordCloud:** Used to generate word clouds for visualizing frequent words.
-   **Streamlit:** Used for making the web app.

**Dataset Used:**
I have used [SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset) downloaded from Kaggle.
It contains one set of SMS messages in English of 5,574 messages, tagged acording being ham (legitimate) or spam.

## Project setup

Clone the project
```bash
git clone https://github.com/Mr-Atanu-Roy/Spam-SMS-Classification

```

or simply download this project from https://github.com/Mr-Atanu-Roy/BlogZilla-Backend

In project directory Create a virtual environment (env)

```bash
  virtualenv env

```

Activate the virtual environment

For windows:

```bash
  env\Script\activate

```

Install dependencies

```bash
  pip install -r requirements.txt

```

To run the web app locally, run the following command

```bash
  streamlit run .\app\app.py

```
You will be automatically redireced to the web app running in your local server

## About the Model:
After various analysis and evaluation of different algo, metrices, parameter and hyperparameters I have used the following for the best output, which is also for the web app:

- TfidfVectorizer is used with max_features hyperparamer as 3000 for vectorizing the text data.
- Multinomial Naive Bayes Algo is used for classification.
- Achived accuracy: 97.68% and precision: 99.19%

For detailed model evaluation and analysis see: **[Analysis](https://github.com/Mr-Atanu-Roy/Spam-SMS-Classification/blob/master/analysis.md)**


## Other Notes
- I have trained the model in the **SMS_detection.ipynb** file in google colab.
- To experiment or run the file upload the file and **spam.csv** dataset(from dataset folder to google colab). Or you can use jupyter note book.


## Author

-   [@Atanu Roy](https://github.com/Mr-Atanu-Roy)
