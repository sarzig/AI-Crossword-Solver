"""
Author: Sarah

This file builds, evaluates, and applies a machine learning pipeline for classifying crossword clues.
It is fully functional (and very useful!).

Functions:
----------
* drop_clue_and_word(df): 
    Drops "clue" and "Word" columns from a DataFrame if present.

* create_pipeline(): 
    Loads manually classified clues, adds features, trains a RandomForest pipeline, and returns both fitted and 
    unfitted versions.

* save_pipeline(fitted_pipeline): 
    Saves the fitted pipeline to a .joblib file, renaming any existing file as a backup.

* get_clue_classification_ml_pipeline(): 
    Loads the saved pipeline from disk.

* assess_pipeline_performance(clues, unfitted_pipeline): 
    Splits clues data into train/test sets and prints classification metrics using the given (or default) pipeline.

* add_top_class_predictions(df, pipeline, top_n): 
    Adds top-N predicted classes and probabilities to a DataFrame using a fitted pipeline.

* predict_single_clue_from_default_pipeline(clue_text, pipeline, top_n): 
    Predicts classification probabilities for a single clue using the default pipeline.

* predict_clues_df_from_default_pipeline(clues_df, keep_features, pipeline, top_n): 
    Classifies a DataFrame of clues and returns predictions, optionally keeping extracted features.
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
from clue_classification_and_processing.clue_features import add_features, move_feature_columns_to_right_of_df, \
    select_numeric_features
from clue_classification_and_processing.helpers import get_clues_by_class, get_project_root, get_clues_dataframe
from sklearn.base import clone


def drop_clue_and_word(df):
    columns_to_drop = [col for col in ["clue", "Word"] if col in df.columns]
    return df.drop(columns=columns_to_drop, errors="ignore")


def create_pipeline():
    """
    This function loads the manually classified clues, calculates tf-idf-based features,
    and creates an ML pipeline.

    :return: pipeline, clues
    """
    # Load the Excel file
    manually_classed_clues = get_clues_by_class(classification_type="manual_only")

    # Filter rows where "Class" is numeric, None, not a string, or less than length 3
    exclude_rows = (
            manually_classed_clues["Class"].isna() |
            manually_classed_clues["Class"].apply(lambda x: isinstance(x, (int, float))) |
            manually_classed_clues["Class"].apply(lambda x: isinstance(x, str) and len(x.strip()) < 3)
    )

    clues = manually_classed_clues[~exclude_rows]

    # add features
    clues = add_features(clues)

    # Define features and labels
    X = clues.drop(columns=["Class", "Word"])
    y = clues["Class"]

    text_features = ["clue"]
    # numeric_features = [col for col in X.columns if col not in text_features and col != "Word"]

    preprocessor = ColumnTransformer([
        ("tfidf_clue", TfidfVectorizer(analyzer='word', ngram_range=(1, 2), max_features=1000), "clue"),
        ("numeric_features", FunctionTransformer(select_numeric_features, validate=False), X.columns)
    ])

    print("All f_ columns:")
    print([col for col in clues.columns if col.startswith("_f_")])

    print("Any non-numeric values?")
    print(clues[[col for col in clues.columns if col.startswith("_f_")]].dtypes)

    print("Sample rows:")
    print(clues[[col for col in clues.columns if col.startswith("_f_")]].head())

    base_pipeline = Pipeline([
        ("features", preprocessor),
        ("classifier", RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced"))
    ])

    # Fit the pipeline
    fitted_pipeline = base_pipeline
    fitted_pipeline.fit(X, y)

    # Create a second unfitted copy
    unfitted_pipeline = clone(base_pipeline)

    return {"fitted_pipeline": fitted_pipeline,
            "unfitted_pipeline": unfitted_pipeline,
            "clues": clues}


def save_pipeline(fitted_pipeline):
    """
    Given a fitted_pipeline, save it to "clue_classification_ml_pipeline.joblib"
    file within the project data folder.

    :param fitted_pipeline: fitted_pipeline to save, trained on manually classified clues
    :return: nothing
    """

    # Pipeline paths
    save_location = os.path.join(get_project_root(),
                                 "data",
                                 "clue_classification_ml_pipeline",
                                 "clue_classification_ml_pipeline.joblib")

    old_location = os.path.join(get_project_root(),
                                "data",
                                "clue_classification_ml_pipeline",
                                "OLDclue_classification_ml_pipeline.joblib")

    # If file already exists, rename it to OLD_clue_classification_ml_pipeline_.joblib
    if os.path.exists(save_location):
        print(f"Pipeline exists at:\n{save_location}.\n\nRenaming temporarily to:\n{old_location}.\n")
        if os.path.exists(old_location):
            os.remove(old_location)
        os.rename(save_location, old_location)

    # Save it and check path exists
    joblib.dump(fitted_pipeline, save_location)

    # If save_location path exists, delete OLD path
    if os.path.exists(save_location):
        print(f"Pipeline saved to:\n{save_location}.\n\nRemoving:\n{old_location}.")

        if os.path.exists(old_location):
            os.remove(old_location)
    else:
        print(f"\nTried saving pipeline to {save_location},\nbut could not.")
        raise FileNotFoundError("Couldn't save pipeline.")


def get_clue_classification_ml_pipeline():
    """
    Retrieves the default clue classification pipeline saved at
    project_root/data/clue_classification_ml_pipeline.joblib

    :return: pipeline object
    """

    # Pipeline path
    loc = os.path.join(get_project_root(),
                       "data",
                       "clue_classification_ml_pipeline",
                       "clue_classification_ml_pipeline.joblib")

    # load and return
    pipeline = joblib.load(loc)
    return pipeline


def assess_pipeline_performance(clues, unfitted_pipeline: Pipeline = None):
    """
    Given a pipeline and a clues dataFrame containing a column called Class (note
    this should be manually classified class), this tests the pipeline on the actual
    vs. predicted values.

    :param unfitted_pipeline:
    :param clues: clues dataFrame with column "Class"
    :return:
    """

    # If the pipeline is None, just use the default one
    if unfitted_pipeline is None:
        unfitted_pipeline = get_clue_classification_ml_pipeline()["unfitted_pipeline"]

    # Target
    y = clues["Class"]

    # You can select all other columns except target as features
    X = clues.drop(columns=["Class"])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

    unfitted_pipeline.fit(X_train, y_train)

    # Evaluate
    y_prediction = unfitted_pipeline.predict(X_test)

    # make the report (zero_division stops the error for low-frequency classes)
    classification_success_report = classification_report(y_test, y_prediction, zero_division=0)
    print(classification_success_report)

    return classification_success_report


def add_top_class_predictions(df, pipeline, top_n=10):
    """
    GenAI created function to return a probability list instead of a single class

    :param df: dataframe with add_features already done, and a column named "clue"
    :param pipeline: a FITTED pipeline
    :param top_n: The number of class matches which should be returned
    :return: the dataframe with the top classes added, and feature columns moved to the right
    """

    # Get class probabilities
    probabilities = pipeline.predict_proba(df)
    class_labels = pipeline.classes_

    # Zip class labels with probabilities for each row, sort by prob desc, and convert np.float64 to float
    predictions = []
    for row in probabilities:
        label_prob_pairs = list(zip(class_labels, row))
        sorted_pairs = sorted(label_prob_pairs, key=lambda x: x[1], reverse=True)
        top_n_pairs = sorted_pairs[:top_n]
        top_n_pairs_as_floats = [(label, float(prob)) for label, prob in top_n_pairs]

        filtered_pairs = [(label, float(prob)) for label, prob in top_n_pairs if prob >= 0.01]
        predictions.append(filtered_pairs)

    df = df.copy()
    df["Top_Predicted_Classes"] = predictions
    df = move_feature_columns_to_right_of_df(df)  # Clean up the columns

    return df


def predict_single_clue_from_default_pipeline(clue_text, pipeline=None, top_n=10):
    """
    Simply is a wrapper for predict_clues_df_from_default_pipeline that does
    clue classification on a single text input. Pipeline should be passed as an argument,
    otherwise the pipeline will be manually loaded every invocation of this function.

    :param clue_text: text to classify
    :param pipeline: pipeline to pass. If none, a pipeline will be loaded
    :return: the classification dictionary of probabilities
    """

    # Create a mini, 1-row dataframe to pass
    mini_df = pd.DataFrame({"clue": [clue_text]})

    # If pipeline=None, a default pipeline is loaded inside
    # predict_clues_df_from_default_pipeline (like: pipeline = get_clue_classification_ml_pipeline())
    classification_df = predict_clues_df_from_default_pipeline(clues_df=mini_df,
                                                               pipeline=pipeline,
                                                               keep_features=False,
                                                               top_n=top_n)

    class_predictions = classification_df.loc[0, "Top_Predicted_Classes"]

    return class_predictions


def predict_clues_df_from_default_pipeline(clues_df, keep_features=False, pipeline=None, top_n=10):
    """
    Given the default saved fitted pipeline, take the clues_df from the input (which
    will be modified to only contain "Clue" and "Word"), apply clues feature addition,
    and then return a dataframe with those predictions added.

    :return:
    """

    # Save original DataFrame
    original_df = clues_df.copy()

    # If "Clue" is in columns, rename to "clue"
    if "Clue" in clues_df.columns:
        clues_df.rename(columns={'Clue': 'clue'}, inplace=True)

    if "clue" not in clues_df.columns:
        print("columns found in input clues_df:")
        print(clues_df.columns)
        raise ValueError("Input DataFrame must contain a 'Clue' or 'clue' column.")

    # Remove extraneous columns in clues_df
    base_cols = ["clue", "Word"]
    minimal_df = clues_df[[col for col in base_cols if col in clues_df.columns]].copy()

    # Fetch the saved pipeline
    pipeline = get_clue_classification_ml_pipeline()

    # Inspect the classifications of manually_classed_clues
    minimal_df = add_features(minimal_df)  # Add the features
    minimal_df = add_top_class_predictions(minimal_df, pipeline, top_n=top_n)

    # Combine back with original
    combined_df = original_df.copy()
    combined_df["Top_Predicted_Classes"] = minimal_df["Top_Predicted_Classes"]

    if keep_features is True:
        combined_df = pd.concat([combined_df, minimal_df[[col for col in minimal_df.columns if col.startswith("_f_")]]], axis=1)
        combined_df = move_feature_columns_to_right_of_df(combined_df)
    else:
        # Drop all feature columns starting with "_f_"
        combined_df = combined_df[[col for col in combined_df.columns if not col.startswith("_f_")]]

    return combined_df


def create_all_clues_predicted():
    all_clues = get_clues_dataframe(delete_dupes=True)
    all_clues_predicted = predict_clues_df_from_default_pipeline(clues_df=all_clues,
                                                                 keep_features=True,
                                                                 top_n=5)
    save_path = os.path.join(get_project_root(),
                             "data",
                             "clue_classification_ml_pipeline",
                             "all_clues_predicted.csv")

    all_clues_predicted.to_csv(save_path)


'''
# Create the pipeline - only needs to be done once per update to manual classifications
# OR to changes to add_features

pipeline_result = create_pipeline()
fitted_pipeline = pipeline_result["fitted_pipeline"]
unfitted_pipeline = pipeline_result["unfitted_pipeline"]
my_clues = pipeline_result["clues"]

# Save the fitted version of the pipeline
save_pipeline(fitted_pipeline)

# Make a report
report = assess_pipeline_performance(clues=my_clues, unfitted_pipeline=unfitted_pipeline)

# Get the saved pipeline
my_pipeline = get_clue_classification_ml_pipeline()



# Inspect the classifications of manually_classed_clues
manually_classed_clues = get_clues_by_class(classification_type="manual_only")  # Get the clues set
manually_classed_clues_predicted = predict_clues_df_from_default_pipeline(clues_df=manually_classed_clues,
                                                                keep_features=False,
                                                                top_n=3)

# Apply this to the actual full dataset!!
all_clues = get_clues_dataframe(delete_dupes=True)
all_clues_predicted = predict_clues_df_from_default_pipeline(clues_df=all_clues,
                                                keep_features=True,
                                                    top_n=3)

all_clues_predicted.to_csv("all clues predicted.csv")

clue_subset = all_clues_predicted[
    (all_clues_predicted['_f_number words'] > 5) &
    (all_clues_predicted['_f_percentage words that are upper-case'] > .3) &
    (all_clues_predicted['_f_ends in question'] == True)

]
'''

