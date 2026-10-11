import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.pyplot as plt

df = pd.read_csv('dataset_vocales_fft.csv')
X = df.drop('label', axis=1).values
y = df['label'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train_scaled, y_train)
acc = clf.score(X_test_scaled, y_test)
print(f"Precisión: {acc*100:.2f}%")

# Guarda scaler y modelo
#joblib.dump(scaler, "scaler_vocales_fft.pkl")
#joblib.dump(clf, "modelo_vocales_fft.pkl")

# Matriz de confusión
y_pred = clf.predict(X_test_scaled)
cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.title('Matriz de confusión')
plt.show()

print(classification_report(y_test, y_pred))
