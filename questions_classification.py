import pandas as pd
from snorkel.labeling import LabelingFunction, labeling_function
from snorkel.labeling import PandasLFApplier
from snorkel.labeling import LFAnalysis
from snorkel.labeling.model import MajorityLabelVoter
from snorkel.labeling.model import LabelModel
from snorkel.utils import probs_to_preds
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression


# load and preprocess training data
def load_training_dataset():

    filename = "comments_train.csv"

    dfs = []
    df = pd.read_csv(filename)
    # Lowercase column names
    df.columns = map(str.lower, df.columns)
    # Remove comment_id field
    df = df.drop("video id", axis=1)
    # Rename fields
    df = df.rename(columns={"title": "video", "comment": "text"})
    # Shuffle order
    df = df.sample(frac=1, random_state=123).reset_index(drop=True)
    dfs.append(df)

    df_train = pd.concat(dfs)

    return df_train


# load and preprocess testing data
def load_test_dataset():

    filename = "comments_test.csv"

    dfs = []
    df = pd.read_csv(filename)
    # Lowercase column names
    df.columns = map(str.lower, df.columns)
    # Remove comment_id field
    df = df.drop("video id", axis=1)
    # Rename fields
    df = df.rename(columns={"title": "video", "comment": "text"})
    # Shuffle order
    df = df.sample(frac=1, random_state=123).reset_index(drop=True)
    dfs.append(df)

    df_test = pd.concat(dfs)

    return df_test


data_train = load_training_dataset()
data_test = load_test_dataset()

# constants to represent the class labels for question, non_question and abstaining.
ABSTAIN = -1
NON_QUESTION = 0
QUESTION = 1


# exploring the training set for initial ideas
data_demo = data_train[["video", "text"]].sample(20, random_state=2)
print(data_demo)


# write an LF to identify question comments using the question mark
@labeling_function()
def question_mark(x):
    if "?" in x.text:
        return QUESTION
    else:
        return ABSTAIN


# write an LF to identify question comments using the question words
@labeling_function()
def question_word(x):
    # 'wh-' question words, e.g., who, what, when, where, which
    if re.search(r"^wh?.{1,5}", x.text.lower(), flags=re.I):
        return QUESTION
    # auxiliary verbs, e.g., do, did, does
    elif re.search(r"^(do|did|does)", x.text.lower(), flags=re.I):
        return QUESTION
    # be-verbs, e.g., is, am, are
    elif re.search(r"^(is|am|are)", x.text.lower(), flags=re.I):
        return QUESTION
    # modal verbs, e.g., can, could
    elif re.search(r"^(can|could)", x.text.lower(), flags=re.I):
        return QUESTION
    else:
        return ABSTAIN


# write Keyword LFs
def keyword_lookup(x, keywords, label):
    for word in keywords:
        if word in x.text.lower():
            return label
        return ABSTAIN


def make_keyword_lf(keywords, label=QUESTION):
    return LabelingFunction(
        name=f"keyword_{keywords[0]}",
        f=keyword_lookup,
        resources=dict(keywords=keywords, label=label),
    )


# list german question words appearing common in the comments
keyword_w = make_keyword_lf(keywords=["wie", "warum", "wer", "wann"])

keyword_k = make_keyword_lf(keywords=["kann", "können", "könnten"])

keyword_sein = make_keyword_lf(keywords=["bin", "bist", "ist", "seid", "sind"])

keyword_verb = make_keyword_lf(keywords=["gibt", "machen", "haben", "habt", "hat"])

# Combining Labeling Function Outputs
lfs = [
    question_mark,
    question_word,
    keyword_w,
    keyword_k,
    keyword_sein,
    keyword_verb
]

# apply label functions
applier = PandasLFApplier(lfs=lfs)
# create a label matrix for the training set
L_train = applier.apply(df=data_train)
# create a label matrix for the test det
L_test = applier.apply(df=data_test)

# summary statistics for the LFs
lf_summary = LFAnalysis(L=L_train, lfs=lfs).lf_summary()
print(lf_summary)

# take the majority vote on a per-data point basis
majority_model = MajorityLabelVoter()
preds_train = majority_model.predict(L=L_train)

# use LabelModel to produce training labels
label_model = LabelModel(cardinality=2, verbose=True)
label_model.fit(L_train=L_train, n_epochs=500, log_freq=100, seed=123)

# result using majority-vote model
Y_test = data_test.label.values
majority_acc = majority_model.score(L=L_test, Y=Y_test, tie_break_policy="random")[
    "accuracy"
]
print(f"{'Majority Vote Accuracy:':<25} {majority_acc * 100:.1f}%")


# results using label model
label_model_acc = label_model.score(L=L_test, Y=Y_test, tie_break_policy="random")[
    "accuracy"
]
print(f"{'Label Model Accuracy:':<25} {label_model_acc * 100:.1f}%")

# representing each data point using "bag of n-gram" feature
probs_train = label_model.predict_proba(L=L_train)
vectorizer = CountVectorizer(ngram_range=(1, 5))
X_train = vectorizer.fit_transform(data_train.text.tolist())
X_test = vectorizer.transform(data_test.text.tolist())

# replace each label distribution with the label having maximum probability
preds_train = probs_to_preds(probs=probs_train)

# train a Scikit-Learn classifier
sklearn_model = LogisticRegression(C=1e3, solver="liblinear")
sklearn_model.fit(X=X_train, y=preds_train)
# result of the classifier accuracy
print(f"Test Accuracy: {sklearn_model.score(X=X_test, y=Y_test) * 100:.1f}%")
