import numpy as np
import pandas as pd

np.random.seed(42)
n = 100000

data = pd.DataFrame({
    "study_hours": np.random.uniform(0, 10, n),
    "attendance": np.random.uniform(40, 100, n),
    "previous_marks": np.random.uniform(0, 100, n),
    "sleep_hours": np.random.uniform(3, 10, n),
    "assignment_score": np.random.uniform(0, 100, n),
    "internet_usage": np.random.uniform(0, 8, n),
})

score = (
    0.8*data["study_hours"] +
    0.05*data["attendance"] +
    0.06*data["previous_marks"] +
    0.05*data["assignment_score"] -
    0.5*data["internet_usage"]
)

# Lower threshold
data["pass"] = (score > 15).astype(int)

print(data["pass"].value_counts())

data.to_csv("student_pass_fail_large.csv", index=False)

print("Dataset Created!")