import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load dataset
df = pd.read_csv(
    r"C:\Users\chand\Downloads\archive\gym_members_exercise_tracking.csv"
)

# Encode Gender
gender_encoder = LabelEncoder()
df['Gender'] = gender_encoder.fit_transform(df['Gender'])

# Encode Workout_Type (Target)
workout_encoder = LabelEncoder()
df['Workout_Type'] = workout_encoder.fit_transform(df['Workout_Type'])

# Features
X = df[
    [
        'Age',
        'Gender',
        'Weight (kg)',
        'Height (m)',
        'BMI',
        'Fat_Percentage',
        'Experience_Level'
    ]
]

# Target
y = df['Workout_Type']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train model
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

# Predictions
pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, pred)

print("Accuracy:", accuracy)

# Save model
joblib.dump(model, "fitfuel_model.pkl")
joblib.dump(gender_encoder, "gender_encoder.pkl")
joblib.dump(workout_encoder, "workout_encoder.pkl")

print("Model Saved Successfully")