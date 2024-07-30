# Model Eğitim ve Değerlendirme

Bu proje, veri setleri üzerinde Karar Ağacı (Decision Tree) algoritması kullanarak model eğitimi ve değerlendirmesi yapmaktadır. Verilerin ön işlenmesi, modelin eğitilmesi, değerlendirilmesi ve sonuçların görselleştirilmesi aşamaları dahil edilmiştir. Bu proje, her tür veri setine uyarlanabilir şekilde tasarlanmıştır.

## Özellikler

- **Veri Ön İşleme**: Kategorik verilerin sayısal verilere dönüştürülmesi ve one-hot encoding işlemleri.
- **Model Eğitimi**: Karar Ağacı algoritması kullanılarak modelin eğitilmesi.
- **Değerlendirme**: Doğruluk, F1 skoru, RMSE, Karmaşıklık Matrisi ve ROC-AUC Eğrisi hesaplamaları ve görselleştirilmesi.

## Kullanım

1. **Veri Dosyasını Hazırlayın**: `emission_dataset.csv` veya benzeri bir veri dosyasını hazır bulundurun. Dosyanızda bağımlı değişken olarak belirleyeceğiniz bir hedef sütun olmalıdır.

2. **Kodunuzu Yapılandırın**: 
   - `file_path` değişkenine veri dosyanızın yolunu girin.
   - `target_column` değişkenine hedef sütunun adını yazın.

3. **Kodunuzu Çalıştırın**: 
   - Öncelikle gerekli kütüphanelerin yüklü olduğundan emin olun.
   - `preprocess_data` fonksiyonunu kullanarak verilerinizi hazırlayın.
   - `train_and_evaluate_model` fonksiyonunu kullanarak modeli eğitin ve değerlendirin.

### Örnek Kullanım

```python
file_path = '/path/to/your/dataset.csv'
target_column = 'your_target_column'  # Hedef sütun adını buraya yazın

X, y = preprocess_data(file_path, target_column)
train_and_evaluate_model(X, y)
