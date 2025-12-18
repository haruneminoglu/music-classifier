# Music Classifier

Bu proje, tek sesli enstrümanlarla çalınan ses kayıtlarından hangi enstrümanın çalındığını otomatik olarak tespit etmeyi amaçlamaktadır. Projede üç temel yaklaşım uygulanmıştır:

- Geleneksel makine öğrenimi (Random Forest)  
- Transfer Learning (YAMNet tabanlı)  
- AutoML (AutoGluon)

---

## 1. Random Forest Modelinin İş Akışı

**Ön İşleme:**  
- Ses dosyaları yüklenir ve normalleştirilir  
- Gürültü azaltma uygulanır  
- Sessizlik kırpma ile gereksiz bölümler temizlenir

**Özellik Çıkarımı:**  
- MFCC, spektral özellikler, chroma gibi toplam 100+ ses özelliği elde edilir

**Veri Seti Oluşturma:**  
- Veri artırma (augmentation) uygulanır  
- Train-test-validation ayrımı yapılır  
- Random Forest için kullanılabilir bir özellik matrisi oluşturulur

**Model Eğitimi:**  
- Hazırlanan özellik tabanlı veri seti kullanılarak Random Forest modeli eğitilir  
- Performans metrikleri değerlendirilir

---

## 2. YAMNet Modelinin İş Akışı

**Ön İşleme:**  
- Ses örnekleme hızı YAMNet standardına uygun olarak 16 kHz'e dönüştürülür

**Özellik Çıkarımı:**  
- YAMNet tarafından her ses dosyasından 1024 boyutlu özellik vektörleri (embeddings) çıkarılır

**Model Mimarisi:**  
- YAMNet katmanları dondurulur (frozen)  
- Çıkarılan embeddings üzerine custom classification head eklenir  
- 5 enstrüman için sınıflandırma yapılır

---

## 3. AutoML Modelinin İş Akışı

**Veri Hazırlama:**  
- Random Forest ve YAMNet için çıkarılmış .pkl formatındaki veri setleri  
- AutoGluon’un kullanabileceği pandas DataFrame’e dönüştürülür

**Model Eğitimi:**  
- Veri setleri AutoGluon’a verilir  
- Birden fazla ML algoritması otomatik olarak denenir  
- Hiperparametre optimizasyonu ve model ensemble işlemleri ile en yüksek performans sağlanır

---





