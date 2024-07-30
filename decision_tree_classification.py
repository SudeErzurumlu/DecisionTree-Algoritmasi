import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, classification_report, roc_curve, auc, mean_squared_error
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns

def preprocess_data(file_path, target_column):
    # Veriyi Pandas ile yükleyelim
    data = pd.read_csv(file_path)
    
    # Verinin ilk birkaç satırını ve sütun bilgilerini kontrol edelim
    print("Veri Başlangıcı:")
    print(data.head())
    print("\nVeri Bilgisi:")
    print(data.info())

    # Bağımsız ve bağımlı değişkenleri ayıralım
    X = data.drop(target_column, axis=1)
    y = data[target_column]

    # Kategorik verileri sayısal verilere dönüştürelim
    categorical_columns = X.select_dtypes(include=['object']).columns
    X_encoded = pd.get_dummies(X, columns=categorical_columns, drop_first=True)

    return X_encoded, y

def train_and_evaluate_model(X, y, test_size=0.3, random_state=42):
    # Eğitim ve test setlerine ayıralım
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    print(f"Eğitim veri seti boyutu: {X_train.shape}")
    print(f"Test veri seti boyutu: {X_test.shape}")

    # Karar ağacı modelini tanımlama
    model = DecisionTreeClassifier(random_state=random_state)

    # Modeli eğitme
    model.fit(X_train, y_train)

    # Test veri seti üzerinde tahmin yapma
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)  # ROC eğrisi için olasılık tahminleri

    # Doğruluk değerini hesaplama
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nDoğruluk değeri: {accuracy}")

    # Karmaşıklık matrisini hesaplama ve normalize etme
    cm = confusion_matrix(y_test, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print("Karmaşıklık Matrisi:")
    print(cm)

    # F1 skorunu hesaplama
    f1 = f1_score(y_test, y_pred, average='weighted')
    print(f"F1 Skoru: {f1}")

    # RMSE'yi hesaplama
    y_test_numeric = y_test.astype('category').cat.codes
    y_pred_numeric = pd.Series(y_pred).astype('category').cat.codes
    rmse = np.sqrt(mean_squared_error(y_test_numeric, y_pred_numeric))
    print(f"RMSE: {rmse}")

    # ROC-AUC Eğrisi için gerekli hesaplamalar
    fpr = {}
    tpr = {}
    roc_auc = {}
    
    y_test_bin = label_binarize(y_test, classes=model.classes_)

    for i in range(len(model.classes_)):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), y_pred_proba.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=model.classes_))

    # Görselleştirmeler
    plt.figure(figsize=(20, 15))

    # 1. Karmaşıklık Matrisi (Normalized)
    plt.subplot(2, 2, 1)
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
    plt.xlabel('Tahmin Edilen')
    plt.ylabel('Gerçek')
    plt.title('Normalize Edilmiş Karmaşıklık Matrisi')

    # 2. Doğruluk, F1 Skoru ve RMSE
    plt.subplot(2, 2, 2)
    metrics = {'Doğruluk': accuracy, 'F1 Skoru': f1, 'RMSE': rmse}
    names = list(metrics.keys())
    values = list(metrics.values())
    plt.bar(names, values, color=['blue', 'green', 'red'])
    plt.title('Model Performans Metrikleri')
    plt.xlabel('Metrik')
    plt.ylabel('Değer')
    plt.ylim(0, 1)
    for i, v in enumerate(values):
        plt.text(i, v + 0.02, f"{v:.2f}", ha='center', fontweight='bold')

    # 3. ROC-AUC Eğrisi
    plt.subplot(2, 2, 3)
    for i in range(len(model.classes_)):
        plt.plot(fpr[i], tpr[i], lw=2, label=f'ROC Sınıf {model.classes_[i]} (AUC = {roc_auc[i]:.2f})')
    plt.plot(fpr["micro"], tpr["micro"], color='deeppink', linestyle=':', linewidth=4, label=f'Mikro Ortalama ROC (AUC = {roc_auc["micro"]:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Yanlış Pozitif Oranı')
    plt.ylabel('Doğru Pozitif Oranı')
    plt.title('ROC-AUC Eğrisi')
    plt.legend(loc="lower right")
    plt.grid(False)

    # 4. Önemli Özellikler
    plt.subplot(2, 2, 4)
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    features = X.columns
    plt.title('Özelliklerin Önemi')
    plt.bar(range(X.shape[1]), importances[indices], align='center')
    plt.xticks(range(X.shape[1]), features[indices], rotation=90)
    plt.xlim([-1, X.shape[1]])

    plt.tight_layout()
    plt.show()

# Kullanım örneği
file_path = 'buraya csv dosyanızın yolunu ekleyin'
target_column = 'category'  # Hedef sütun adını buraya yazın

X, y = preprocess_data(file_path, target_column)
train_and_evaluate_model(X, y)
