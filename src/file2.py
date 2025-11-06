import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import dagshub 

# ✅ Initialize DagsHub MLflow tracking
dagshub.init(repo_owner='TheAnalyst07',
             repo_name='MLflow-experiment-tracking',
             mlflow=True)

# ✅ Load Data
data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.10, random_state=42
)

# ✅ Set Experiment Name
mlflow.set_experiment("iris_experiment")

max_depth = 10
n_estimators = 6

with mlflow.start_run():
    # Train model
    model = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators, random_state=42)
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    # ✅ Log params correctly
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)

    # ✅ Log metrics
    mlflow.log_metric("accuracy", acc)
    
    print("Accuracy:", acc)

    # ✅ Log model
    mlflow.sklearn.log_model(model, "model")

    # ✅ Confusion matrix
    cm = confusion_matrix(y_test, preds)
    plt.figure(figsize=(8,8))
    sns.heatmap(cm, annot=True, fmt=".1f", cmap='Blues')
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix | Accuracy: {acc:.3f}")
    plt.savefig("confusion_matrix.png")

    # ✅ Log artifacts
    mlflow.log_artifact("confusion_matrix.png")
    mlflow.log_artifact(__file__)

    # ✅ Tags
    mlflow.set_tag("Author", "Rahul")
    mlflow.set_tag("Project", "MLflow+DagsHub")
