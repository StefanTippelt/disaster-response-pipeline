# -*- coding: utf-8 -*-
import sys
from sqlalchemy import create_engine
import pandas as pd
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

import pickle

from nltk import download
from nltk.tokenize import word_tokenize
from sklearn.metrics import classification_report

# download nltk dependencies
download("punkt")
download("wordnet")


def load_data(database_filepath: str):
    #  -> tuple[pd.Series, pd.DataFrame, list]
    """Load data from database."""
    engine = create_engine("sqlite:///" + database_filepath)
    df = pd.read_sql_table("disaster_messages", con=engine)
    X = df["message"]
    Y = df.iloc[:, 4:]
    category_names = Y.columns
    return X, Y, category_names


def tokenize(text: str) -> list:
    return word_tokenize(text)


def build_model():
    """Build ML model using pipeline and grid search."""
    pipeline = Pipeline(
        [
            ("vect", CountVectorizer(tokenizer=tokenize)),
            ("tfidf", TfidfTransformer()),
            ("clf", MultiOutputClassifier(RandomForestClassifier())),
        ]
    )

    parameters = {
        "clf__estimator__min_samples_split": [
            5
            # , 10, 15
        ],
        "clf__estimator__n_estimators": [
            50
            # , 100, 150
        ],
    }

    cv = GridSearchCV(
        pipeline,
        param_grid=parameters,
        scoring="accuracy",
        verbose=1,
        n_jobs=-1,
    )
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """Evaluate performance of model."""
    y_pred = model.predict(X_test)
    print(
        classification_report(
            y_true=Y_test, y_pred=y_pred, target_names=category_names
        )
    )
    return None


def save_model(model, model_filepath):
    """Save trained model to database."""
    with open(model_filepath, "wb") as file:
        pickle.dump(model, file)
    return None


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print(f"Loading data...\n    DATABASE: {database_filepath}")
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=0.2
        )

        print("Building model...")
        model = build_model()

        print("Training model...")
        model.fit(X_train, Y_train)

        print("Evaluating model...")
        evaluate_model(model, X_test, Y_test, category_names)

        print(f"Saving model...\n    MODEL: {model_filepath}")
        save_model(model, model_filepath)

        print("Trained model saved!")

    else:
        print(
            "Please provide the filepath of the disaster messages database "
            "as the first argument and the filepath of the pickle file to "
            "save the model to as the second argument. \n\nExample: python "
            "train_classifier.py ../data/disaster.db classifier.pkl"
        )


if __name__ == "__main__":
    main()
