# -*- coding: utf-8 -*-
import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(
    messages_filepath: str, categories_filepath: str
) -> pd.DataFrame:
    """Load messages and categories merged into a single dataframe."""
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    categories = categories.set_index("id")
    messages = messages.set_index("id")

    # Merge the messages and categories datasets using the common id
    df = pd.DataFrame.join(self=messages, other=categories)

    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean dataframe by splitting categories, dropping processed columns and
    removing duplicates.
    """
    # create a dataframe of the 36 individual category columns
    categories = df["categories"].str.split(";", expand=True)

    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything
    # up to the second to last character of each string with slicing
    category_colnames = [x.split("-")[0] for x in categories.iloc[0].tolist()]

    # rename the columns of `categories`
    categories.columns = category_colnames
    # set each value based on the last character of the string
    for column in categories:
        categories[column] = [x[1] for x in categories[column].str.split("-")]
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])

    # drop the original categories column from `df`
    df = df.drop(columns=["categories"])
    df = pd.DataFrame.join(self=df, other=categories)
    # drop duplicates
    df = df.drop_duplicates()
    return df


def save_data(df: pd.DataFrame, database_filename: str) -> None:
    """Save dataframe to database."""
    engine = create_engine("sqlite:///" + database_filename)
    df.to_sql("disaster_messages", engine, index=False)
    return None


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[
            1:
        ]

        print(
            "Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}".format(
                messages_filepath, categories_filepath
            )
        )
        df = load_data(messages_filepath, categories_filepath)

        print("Cleaning data...")
        df = clean_data(df)

        print("Saving data...\n    DATABASE: {}".format(database_filepath))
        save_data(df, database_filepath)

        print("Cleaned data saved to database!")

    else:
        print(
            "Please provide the filepaths of the messages and categories "
            "datasets as the first and second argument respectively, as "
            "well as the filepath of the database to save the cleaned data "
            "to as the third argument. \n\nExample: python process_data.py "
            "disaster_messages.csv disaster_categories.csv "
            "disaster.db"
        )


if __name__ == "__main__":
    main()
