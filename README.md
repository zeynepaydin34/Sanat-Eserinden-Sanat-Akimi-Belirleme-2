# Görüntü Sınıflandırma Modelleri: DeVit, LeVit, DINO, CAIT ve MobileVit

Bu proje, **DeVit**, **LeVit**, **DINO**, **CAIT** ve **MobileVit** gibi gelişmiş görsel modeller kullanarak görüntü sınıflandırma görevini gerçekleştirmektedir. Proje, modellerin kayıp (loss), doğruluk (accuracy), karmaşıklık matrisi (confusion matrix), ROC eğrisi ve AUC gibi metriklerle değerlendirilmesini sağlar.

## Proje Özellikleri

### 1. Model Seçimi
Proje, görüntü sınıflandırma için popüler derin öğrenme modelleri arasında yer alan **DeVit**, **LeVit**, **DINO**, **CAIT** ve **MobileVit**'i kullanmaktadır. Bu modeller, özellikle derin öğrenme ve görsel tanıma alanlarında yüksek başarılar elde etmiş önceden eğitilmiş model ağırlıklarına sahiptir ve `timm` kütüphanesi ile kolayca erişilebilir.

Her bir modelin özellikleri:
- **DeVit**: Transformer tabanlı bir model olup, görsel veri üzerinde etkili performans gösterir.
- **LeVit**: Daha hafif ve verimli bir model olup, özellikle mobil cihazlar için optimize edilmiştir.
- **DINO**: Self-supervised (denetimsiz) öğrenme yaklaşımını kullanan bir modeldir.
- **CAIT**: Vision Transformer (ViT) tabanlı bir model olup, görsel görevlerde yüksek doğruluk oranları sağlar.
- **MobileVit**: Mobil cihazlarda çalıştırılabilirlik açısından optimize edilmiş, düşük bellek ve hesaplama gereksinimlerine sahip bir modeldir.

### 2. Metrik Hesaplamaları
Projede kullanılan metrikler, modelin eğitim ve test performansını değerlendirmek için kullanılır. Aşağıda kullanılan metrikler hakkında detaylı bilgiler bulunmaktadır:
- **Eğitim ve Test Kaybı (Loss)**: Eğitim ve test süreçlerinde modelin hata miktarı hesaplanır.
- **Doğruluk (Accuracy)**: Modelin doğru tahmin ettiği örneklerin oranı.
- **Karmaşıklık Matrisi (Confusion Matrix)**: Modelin her sınıf için doğru ve yanlış tahminlerini gösteren bir tablodur.
- **ROC Eğrisi (Receiver Operating Characteristic Curve)**: Modelin doğru pozitif oranı (True Positive Rate) ile yanlış pozitif oranını (False Positive Rate) gösteren eğri.
- **AUC (Area Under Curve)**: ROC eğrisinin altındaki alanın ölçüsüdür.

### 3. Checkpoint Kaydetme
Modelin eğitim sırasında elde edilen ağırlıklar ve parametreler kaydedilebilir. Bu, eğitim sırasında modelin performansını değerlendirmek için önemli bir adımdır. Ayrıca, eğitim sırasında modelin durumu kaydedilerek, eğitim sürecine gerektiğinde devam edilebilecek şekilde checkpoint'ler oluşturulabilir. 

**Checkpoint Kaydetme Fonksiyonu Örneği:**
```python
def save_checkpoint(model, optimizer, epoch, train_loss, test_loss, train_accuracy, test_accuracy, y_true, y_pred, checkpoint_path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'test_loss': test_loss,
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'y_true': y_true,
        'y_pred': y_pred,
    }, checkpoint_path)


