import pandas as pd

num_rounds = 5

rules_suffix = "rules"
class_names = pd.Series(
    [
        "Pregnancies",
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI",
        "DiabetesPedigreeFunction",
        "Age",
    ]
)
