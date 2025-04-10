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

'''
def select_numeric_features(clues):
    """
    Cute little helper function to select the numeric features in clues_df. 
    
    :param clues: the input dataframe, typically the manually classified clues df
                  with enriched columns (with features).
    :return: the subset of clues that is non-numeric
    """
    return clues[[col for col in clues.columns if col not in ["Clue", "Word"]]]


def drop_clue_and_word(df):
    return df.drop(columns=["Clue", "Word"])


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

    X = clues.drop(columns=["Class"])

    text_features = ["Clue"]

    numeric_features = [col for col in X.columns if col not in text_features and col != "Word"]

    # Assume all columns are numeric (boolean, int, float)


    # Apply a TfidfVectorizer to "Clue" column - assesses how the frequency of words predict class inclusion
    # Additionally, convert numeric features to a usable

    preprocessor = ColumnTransformer([
        ("tfidf_clue", TfidfVectorizer(analyzer='word', ngram_range=(1, 2), max_features=1000), "Clue"),
        ("numeric_features", FunctionTransformer(lambda x: x[numeric_features], validate=False), numeric_features)
    ])

    # Build the pipeline
    pipeline = Pipeline([
        ("features", preprocessor),
        ("classifier",
         RandomForestClassifier(n_estimators=100,
                                random_state=42,  # the answer to life, the universe, and everything
                                class_weight="balanced"))  # balanced ensures low frequency classes not lost
    ])

    return pipeline, clues
'''


def drop_clue_and_word(df):
    columns_to_drop = [col for col in ["Clue", "Word"] if col in df.columns]
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

    X = clues.drop(columns=["Class", "Word"])

    text_features = ["Clue"]
    numeric_features = [col for col in X.columns if col not in text_features and col != "Word"]

    preprocessor = ColumnTransformer([
        ("tfidf_clue", TfidfVectorizer(analyzer='word', ngram_range=(1, 2), max_features=1000), "Clue"),
        ("numeric_features", FunctionTransformer(select_numeric_features, validate=False), X.columns)
    ])

    print("All f_ columns:")
    print([col for col in clues.columns if col.startswith("_f_")])

    print("Any non-numeric values?")
    print(clues[[col for col in clues.columns if col.startswith("_f_")]].dtypes)

    print("Sample rows:")
    print(clues[[col for col in clues.columns if col.startswith("_f_")]].head())

    pipeline = Pipeline([
        ("features", preprocessor),
        ("classifier", RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced"))
    ])

    return pipeline, clues


def save_pipeline(pipeline):
    """
    Given a pipeline, save it to "clue_classification_ml_pipeline.joblib"
    file within the project data folder.

    :param pipeline: pipeline to save, trained on manually classified clues
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
    joblib.dump(pipeline, save_location)

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
    :return:
    """
    # Pipeline path
    loc = os.path.join(get_project_root(),
                       "data",
                       "clue_classification_ml_pipeline.joblib")

    # load and return
    pipeline = joblib.load(loc)
    return pipeline


def assess_pipeline_performance(clues, pipeline=None):
    """
    Given a pipeline and a clues dataFrame containing a column called Class (note
    this should be manually classified class), this tests the pipeline on the actual
    vs. predicted values
    :param pipeline: trained pipeline
    :param clues: clues dataFrame with column "Class"
    :return:
    """

    # If the pipeline is None, just use the default one
    if pipeline is None:
        pipeline = get_clue_classification_ml_pipeline()

    # Target
    y = clues["Class"]

    # You can select all other columns except target as features
    X = clues.drop(columns=["Class"])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

    pipeline.fit(X_train, y_train)

    # Evaluate
    y_prediction = pipeline.predict(X_test)

    # make the report (zero_division stops the error for low-frequency classes)
    classification_success_report = classification_report(y_test, y_prediction, zero_division=0)
    print(classification_success_report)

    return classification_success_report


my_pipeline, my_clues = create_pipeline()
save_pipeline(my_pipeline)
report = assess_pipeline_performance(clues=my_clues, pipeline=my_pipeline)

manually_classed_clues = get_clues_by_class(classification_type="manual_only")
manually_classed_clues = add_features(manually_classed_clues)

# Step 3: Predict
predictions = my_pipeline.predict(manually_classed_clues)

# Step 4: Add predictions to DataFrame
manually_classed_clues = manually_classed_clues.copy()
manually_classed_clues["Predicted_Class"] = predictions  # Initialize with None
manually_classed_clues = move_feature_columns_to_right_of_df(manually_classed_clues)

all_clues = get_clues_dataframe(delete_dupes=True)
all_clues = add_features(all_clues)

# Step 3: Predict
predictions = my_pipeline.predict(all_clues)

# Step 4: Add predictions to DataFrame
all_clues = all_clues.copy()
all_clues["Predicted_Class"] = predictions  # Initialize with None
all_clues = move_feature_columns_to_right_of_df(all_clues)
