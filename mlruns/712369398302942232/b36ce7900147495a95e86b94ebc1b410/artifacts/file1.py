import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# Load Data
data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.10, random_state=42)

# Start experiment tracking
mlflow.set_experiment("MLflow Tracing Tutorial")

max_depth = 10
n_estimators = 6
#mention your experment name
mlflow.set_experiment("iris_experiment")

with mlflow.start_run():
    model = RandomForestClassifier(max_depth=max_depth, n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    # Log parameters and metrics
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_metric("accuracy", acc)
    
    # Save model
    mlflow.sklearn.log_model(model, "model")
    print(acc)

    #create a confusing matrix 
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, preds)
    plt.figure(figsize=(9,9))
    sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r');
    plt.ylabel('Actual label');
    plt.xlabel('Predicted label');
    all_sample_title = 'Accuracy Score: {0}'.format(acc)
    plt.title(all_sample_title, size = 15);
    print(cm)
    plt.savefig('confusion_matrix.png')

    mlflow.log_artifact('confusion_matrix.png')
    mlflow.log_artifact(__file__)
    mlflow.set_tag("Author", "Rahul", "Project", "MLflow")
   # log the model
    mlflow.sklearn.log_model(model, "model")
    print(acc)
    
