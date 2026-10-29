import pandas as pd

RAW_PATH = "tourism_project/data/tourism.csv"

# Load the raw dataset. A small number of rows contain an app name with an
# unescaped embedded quote (e.g. Retro Arcade "Classic" Games), which breaks
# CSV tokenization. on_bad_lines="skip" drops just those malformed rows
# instead of failing the whole pipeline; the count skipped is reported below.
rows_before = sum(1 for _ in open(RAW_PATH, encoding="utf-8", errors="replace")) - 1
df = pd.read_csv(RAW_PATH, on_bad_lines="skip")
rows_skipped = rows_before - df.shape[0]
if rows_skipped:
    print(f"Skipped {rows_skipped} malformed row(s) while parsing {RAW_PATH}.")

# Validate that the expected columns are present before registering it
expected_columns = [
    "CustomerID","ProdTaken","Age","TypeofContact","CityTier","DurationOfPitch",
    "Occupation","Gender","NumberOfPersonVisiting","NumberOfFollowups",
    "ProductPitched","PreferredPropertyStar","MaritalStatus","NumberOfTrips",
    "Passport","PitchSatisfactionScore","OwnCar","NumberOfChildrenVisiting",
    "Designation","MonthlyIncome"
]
missing = [c for c in expected_columns if c not in df.columns]
if missing:
    raise ValueError(f"Dataset is missing expected columns: {missing}")

print("Dataset registered successfully.")
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
print("Columns:", list(df.columns))
print("Target column preview (ProdTaken):")
print(df["ProdTaken"].describe())
