# **SPAM SMS CLASSIFICATION**

This project focuses on building a machine learning model to classify SMS messages as either "ham" (legitimate) or "spam".

**Tech Stack:**

-   **Python:** The primary programming language used for the project.
-   **Pandas:** Used for data manipulation and analysis.
-   **NumPy:** Used for numerical operations.
-   **Matplotlib & Seaborn:** Used for data visualization.
-   **NLTK:** Used for natural language processing tasks like tokenization, stop word removal, and stemming.
-   **Scikit-learn:** Used for machine learning model building, including text vectorization (CountVectorizer, TfidfVectorizer), model training (Naive Bayes classifiers), and evaluation metrics.
-   **WordCloud:** Used to generate word clouds for visualizing frequent words.

**Dataset Used:**
https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset
It contains one set of SMS messages in English of 5,574 messages, tagged acording being ham (legitimate) or spam.

## Installation

Create a folder and open terminal and install this project by
command

```bash
git clone https://github.com/Mr-Atanu-Roy/BlogZilla-Backend

```

or simply download this project from https://github.com/Mr-Atanu-Roy/BlogZilla-Backend

In project directory Create a virtual environment of any name(say env)

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

Run migration commands

```bash
  py manage.py makemigrations
  py manage.py migrate

```

Create a super user

```bash
  py manage.py createsuperuser

```

Then add some data into database

To run the project in your localserver

```bash
  py manage.py runserver

```

Now the application is available at: http://127.0.0.1:8000/

To allow request from any end point add the origin to `CORS_ORIGIN_WHITELIST` in settings.py

## Author

-   [@Atanu Roy](https://github.com/Mr-Atanu-Roy)
