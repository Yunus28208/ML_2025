# ==== MODEL MACHINE LEARNING ====
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score

# Load dataset
df = pd.read_csv('/content/plat_kendaraan_dataset.csv')  # Sesuaikan path jika upload di Colab

# Tampilkan 5 data pertama
print("\n=== Contoh Data ===")
print(df.head())

# Pisahkan fitur dan label
X = df['plat']
y = df['kategori']

# Vektorisasi teks plat nomor
vectorizer = CountVectorizer()
X_vec = vectorizer.fit_transform(X)

# Split data: 80% latih, 20% uji
X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)

# Model Naive Bayes
model = MultinomialNB()
model.fit(X_train, y_train)

# Prediksi
y_pred = model.predict(X_test)

# Evaluasi
print("\n=== Evaluasi Model ===")
print("Akurasi:", accuracy_score(y_test, y_pred))
print("\nLaporan Klasifikasi:\n", classification_report(y_test, y_pred))

# Contoh hasil prediksi
print("\n=== Contoh Hasil Prediksi ===")
for i in range(10):
    print(f"Plat: {X.iloc[i]} | Asli: {y.iloc[i]} | Prediksi: {model.predict(vectorizer.transform([X.iloc[i]]))[0]}")
