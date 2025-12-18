"""
AutoML Classifier - AutoGluon ile otomatik model eğitimi ve tahmin
DataManager YAMNet embeddings ve Traditional Features desteği
Proje yapısı: src/ml_models/automl_classifier.py

Özellikler:
- Eğitim modu: YAMNet veya Traditional Features ile model eğitimi
- Tahmin modu: Eğitilmiş model ile yeni ses dosyalarını tahmin etme
"""

import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional, Literal, List
from autogluon.tabular import TabularDataset, TabularPredictor
import sys
import os
import librosa
import tensorflow as tf
import tensorflow_hub as hub
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# Proje kök dizinini sys.path'e ekle
current_dir = os.path.dirname(os.path.abspath(__file__))  # ml_models/
src_dir = os.path.dirname(current_dir)  # src/
project_root = os.path.dirname(src_dir)  # proje kökü
sys.path.insert(0, project_root)


class AutoMLClassifier:
    """
    DataManager çıktılarını AutoGluon ile eğiten classifier
    YAMNet embeddings ve Traditional Features için optimize edilmiş
    """

    def __init__(self, project_root: str = None):
        """
        Args:
            project_root: Proje kök dizini (None ise otomatik tespit)
        """
        if project_root is None:
            # Otomatik tespit: src/ml_models/ -> proje kökü
            self.project_root = Path(__file__).parent.parent.parent
        else:
            self.project_root = Path(project_root)

        self.data_dir = self.project_root / "data" / "processed" / "datasets"
        self.models_dir = self.project_root / "models"

        # YAMNet model cache
        self.yamnet_model = None

        print(f"🤖 AutoMLClassifier hazır")
        print(f"📁 Proje kökü: {self.project_root}")
        print(f"📁 Dataset dizini: {self.data_dir}")
        print(f"📁 Models dizini: {self.models_dir}")

    def load_dataset(self, pkl_path: str) -> Dict:
        """Pickle dataset'i yükler ve tipini kontrol eder"""
        pkl_path = Path(pkl_path)

        if not pkl_path.exists():
            raise FileNotFoundError(f"❌ Dataset bulunamadı: {pkl_path}")

        with open(pkl_path, "rb") as f:
            dataset = pickle.load(f)

        # Dataset tipini kontrol et
        if "embeddings" in dataset:
            data_type = "yamnet"
            embedding_type = dataset.get("embedding_type", "unknown")
            print(f"✅ YAMNet dataset yüklendi (type: {embedding_type})")
        elif "features" in dataset:
            data_type = "traditional"
            print(f"✅ Traditional features dataset yüklendi")
        else:
            raise ValueError("❌ Dataset'te 'embeddings' veya 'features' bulunamadı!")

        dataset["_detected_type"] = data_type
        return dataset

    def convert_to_dataframe(
        self, dataset_dict: Dict, label_column: str = "instrument"
    ) -> pd.DataFrame:
        """
        DataManager formatını AutoGluon DataFrame'e dönüştürür

        Args:
            dataset_dict: DataManager pickle çıktısı
            label_column: Label sütun ismi

        Returns:
            pd.DataFrame: AutoGluon uyumlu format
        """
        data_type = dataset_dict.get("_detected_type", "unknown")

        if data_type == "yamnet":
            # YAMNet embeddings (numpy array)
            data = dataset_dict["embeddings"]

            if not isinstance(data, np.ndarray):
                data = np.array(data)

            print(f"📊 YAMNet embeddings dönüştürülüyor...")
            print(f"   Shape: {data.shape}")
            print(f"   Dtype: {data.dtype}")

            # Sütun isimleri: embedding_0, embedding_1, ..., embedding_1023
            columns = [f"embedding_{i}" for i in range(data.shape[1])]

        elif data_type == "traditional":
            # Traditional features (list of dicts veya numpy array)
            features = dataset_dict["features"]

            if isinstance(features, list) and len(features) > 0:
                if isinstance(features[0], dict):
                    # List of dicts -> numpy array
                    data = np.array([list(f.values()) for f in features])
                    print(f"📊 Traditional features (dict) dönüştürülüyor...")
                else:
                    # Zaten array/list
                    data = np.array(features)
                    print(f"📊 Traditional features (array) dönüştürülüyor...")
            else:
                data = np.array(features)

            print(f"   Shape: {data.shape}")
            print(f"   Dtype: {data.dtype}")

            # Sütun isimleri: feature_0, feature_1, ...
            columns = [f"feature_{i}" for i in range(data.shape[1])]

        else:
            raise ValueError(f"❌ Bilinmeyen data type: {data_type}")

        # Labels
        labels = dataset_dict["labels"]

        if len(data) != len(labels):
            raise ValueError(
                f"❌ Data-label boyut uyumsuzluğu: {len(data)} vs {len(labels)}"
            )

        # DataFrame oluştur
        df = pd.DataFrame(data, columns=columns)
        df[label_column] = labels

        print(f"✅ DataFrame oluşturuldu:")
        print(f"   Satırlar: {len(df)}")
        print(f"   Feature sütunlar: {len(columns)}")
        print(f"   Sınıf dağılımı:")
        print(df[label_column].value_counts().to_string())

        return df

    def prepare_datasets(
        self,
        dataset_type: Literal["yamnet", "traditional"],
        save_csv: bool = True,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Train/Val/Test setlerini AutoGluon formatına hazırlar

        Args:
            dataset_type: "yamnet" veya "traditional"
            save_csv: CSV olarak kaydet mi?

        Returns:
            Tuple[train_df, val_df, test_df]
        """
        print("=" * 70)
        print(f"🔄 {dataset_type.upper()} veri setleri hazırlanıyor...")
        print("=" * 70)

        # Dataset yollarını ayarla
        if dataset_type == "yamnet":
            # YAMNet için 16kHz dataset'leri
            train_pkl = self.data_dir / "good_sounds_yamnet_train_dataset.pkl"
            val_pkl = self.data_dir / "good_sounds_yamnet_val_dataset.pkl"
            test_pkl = self.data_dir / "good_sounds_yamnet_test_dataset.pkl"
            csv_dir = self.project_root / "data" / "processed" / "autogluon_yamnet"

        elif dataset_type == "traditional":
            # Traditional features için 22kHz dataset'leri
            train_pkl = self.data_dir / "good_sounds_features_train_dataset.pkl"
            val_pkl = self.data_dir / "good_sounds_features_val_dataset.pkl"
            test_pkl = self.data_dir / "good_sounds_features_test_dataset.pkl"
            csv_dir = self.project_root / "data" / "processed" / "autogluon_features"

        else:
            raise ValueError(f"❌ Geçersiz dataset_type: {dataset_type}")

        # Dataset'leri yükle
        print("\n1️⃣  Dataset'ler yükleniyor...")
        print(f"   Train: {train_pkl.name}")
        print(f"   Val: {val_pkl.name}")
        print(f"   Test: {test_pkl.name}")

        train_dict = self.load_dataset(str(train_pkl))
        val_dict = self.load_dataset(str(val_pkl))
        test_dict = self.load_dataset(str(test_pkl))

        # DataFrame'lere dönüştür
        print("\n2️⃣  DataFrame'lere dönüştürülüyor...")
        train_df = self.convert_to_dataframe(train_dict, "instrument")
        val_df = self.convert_to_dataframe(val_dict, "instrument")
        test_df = self.convert_to_dataframe(test_dict, "instrument")

        # CSV olarak kaydet (opsiyonel)
        if save_csv:
            print(f"\n3️⃣  CSV dosyaları kaydediliyor...")
            csv_dir.mkdir(parents=True, exist_ok=True)

            train_csv = csv_dir / "train.csv"
            val_csv = csv_dir / "val.csv"
            test_csv = csv_dir / "test.csv"

            train_df.to_csv(train_csv, index=False)
            val_df.to_csv(val_csv, index=False)
            test_df.to_csv(test_csv, index=False)

            print(f"   ✅ Train: {train_csv}")
            print(f"   ✅ Val: {val_csv}")
            print(f"   ✅ Test: {test_csv}")

        print("\n" + "=" * 70)
        print("✅ AutoGluon veri setleri hazır!")
        print("=" * 70)

        return train_df, val_df, test_df

    def train(
        self,
        train_df: pd.DataFrame,
        dataset_type: Literal["yamnet", "traditional"],
        label_column: str = "instrument",
        time_limit: int = 3600,
        eval_metric: str = "accuracy",
        presets: str = "best_quality",
    ) -> TabularPredictor:
        """
        AutoGluon ile model eğitir

        Args:
            train_df: Eğitim DataFrame'i
            dataset_type: "yamnet" veya "traditional"
            label_column: Label sütun ismi
            time_limit: Maksimum eğitim süresi (saniye)
            eval_metric: Değerlendirme metriği
            presets: AutoGluon preset

        Returns:
            TabularPredictor: Eğitilmiş model
        """
        print("=" * 70)
        print(f"🤖 AutoGluon Training - {dataset_type.upper()}")
        print("=" * 70)
        print(f"⏱️  Time limit: {time_limit}s ({time_limit/60:.1f} dakika)")
        print(f"🎯 Preset: {presets}")
        print(f"📊 Eval metric: {eval_metric}")

        # Model kayıt dizini
        output_dir = self.models_dir / f"autogluon_{dataset_type}"
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"📁 Output dir: {output_dir}")

        # TabularPredictor oluştur
        predictor = TabularPredictor(
            label=label_column, eval_metric=eval_metric, path=str(output_dir)
        )

        # Eğitim
        print("\n🚀 Training başladı...")
        predictor.fit(
            train_data=train_df,
            time_limit=time_limit,
            presets=presets,
            verbosity=2,  # Detaylı log
        )

        print("\n✅ Training tamamlandı!")
        print(f"📁 Model kaydedildi: {output_dir}")

        return predictor

    def evaluate(
        self,
        predictor: TabularPredictor,
        test_df: pd.DataFrame,
        label_column: str = "instrument",
    ) -> Dict:
        """
        AutoGluon modelini değerlendirir

        Args:
            predictor: Eğitilmiş AutoGluon model
            test_df: Test DataFrame'i
            label_column: Label sütun ismi

        Returns:
            Dict: Değerlendirme sonuçları
        """
        print("=" * 70)
        print("📊 Model Değerlendirme")
        print("=" * 70)

        # Test seti üzerinde tahmin
        y_pred = predictor.predict(test_df.drop(columns=[label_column]))
        y_true = test_df[label_column]

        # Performans metrikleri
        from sklearn.metrics import (
            accuracy_score,
            precision_recall_fscore_support,
            classification_report,
            confusion_matrix,
        )

        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="weighted"
        )

        results = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "classification_report": classification_report(y_true, y_pred),
            "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
            "y_true": y_true.tolist(),
            "y_pred": y_pred.tolist(),
        }

        print(f"\n📈 Test Sonuçları:")
        print(f"   Accuracy:  {accuracy:.4f}")
        print(f"   Precision: {precision:.4f}")
        print(f"   Recall:    {recall:.4f}")
        print(f"   F1-Score:  {f1:.4f}")

        print(f"\n📋 Detaylı Rapor:")
        print(results["classification_report"])

        # Leaderboard
        print(f"\n🏆 AutoGluon Model Leaderboard:")
        leaderboard = predictor.leaderboard(test_df, silent=True)
        print(leaderboard)

        return results

    def full_pipeline(
        self,
        dataset_type: Literal["yamnet", "traditional"],
        time_limit: int = 1800,
        presets: str = "medium_quality",
    ) -> Dict:
        """
        TEK KOMUTLA: Dönüştür + Eğit + Değerlendir

        Args:
            dataset_type: "yamnet" veya "traditional"
            time_limit: Eğitim süresi (saniye)
            presets: "best_quality", "high_quality", "medium_quality"

        Returns:
            Dict: Tüm sonuçlar
        """
        print("🚀 TAM PIPELINE - Tek Komut")
        print("=" * 70)

        # 1️⃣ Veri setlerini hazırla
        train_df, val_df, test_df = self.prepare_datasets(
            dataset_type=dataset_type, save_csv=True
        )

        # Train + Val birleştir
        combined_train = pd.concat([train_df, val_df], ignore_index=True)
        print(f"\n🔄 Train + Val birleştirildi: {len(combined_train)} örnek")

        # 2️⃣ Model eğit
        predictor = self.train(
            train_df=combined_train,
            dataset_type=dataset_type,
            label_column="instrument",
            time_limit=time_limit,
            presets=presets,
        )

        # 3️⃣ Değerlendir
        results = self.evaluate(predictor=predictor, test_df=test_df)

        # Model bilgilerini ekle
        model_path = self.models_dir / f"autogluon_{dataset_type}"

        output = {
            "predictor": predictor,
            "results": results,
            "model_path": str(model_path),
            "dataset_type": dataset_type,
            "test_accuracy": results["accuracy"],
            "test_f1": results["f1_score"],
        }

        # Özet
        print("\n" + "=" * 70)
        print("✅ PIPELINE TAMAMLANDI!")
        print("=" * 70)
        print(f"📊 Dataset Type: {dataset_type}")
        print(f"📊 Test Accuracy: {results['accuracy']:.4f}")
        print(f"📊 Test F1-Score: {results['f1_score']:.4f}")
        print(f"📁 Model: {model_path}")

        return output

    def list_available_models(self) -> List[Dict[str, str]]:
        """Eğitilmiş modelleri listeler"""
        models = []

        if not self.models_dir.exists():
            print("⚠️ Models dizini bulunamadı!")
            return models

        for model_dir in self.models_dir.iterdir():
            if model_dir.is_dir() and model_dir.name.startswith("autogluon_"):
                # Model tipini tespit et (iyileştirilmiş)
                if "yamnet" in model_dir.name:
                    model_type = "yamnet"
                elif "traditional" in model_dir.name or "features" in model_dir.name:
                    model_type = "traditional"
                else:
                    model_type = "unknown"

                models.append(
                    {
                        "name": model_dir.name,
                        "path": str(model_dir),
                        "type": model_type,
                    }
                )

        return models

    def load_yamnet_model(self):
        """YAMNet modelini yükler (tahmin için)"""
        if self.yamnet_model is None:
            try:
                print("📥 YAMNet modeli yükleniyor...")
                self.yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")

                # Test
                test_audio = np.zeros(16000, dtype=np.float32)
                _, embeddings, _ = self.yamnet_model(test_audio)
                print(f"✅ YAMNet yüklendi! Embedding shape: {embeddings.shape}")

                return True
            except Exception as e:
                print(f"❌ YAMNet yükleme hatası: {e}")
                return False
        return True

    def extract_yamnet_embedding(self, audio_path: str) -> np.ndarray:
        """
        Ses dosyasından YAMNet embedding çıkarır

        Args:
            audio_path: Ses dosyası yolu

        Returns:
            np.ndarray: 1024-dim embedding
        """
        if not self.load_yamnet_model():
            raise RuntimeError("YAMNet modeli yüklenemedi!")

        # 16kHz'de yükle
        waveform, _ = librosa.load(audio_path, sr=16000, mono=True)

        # Normalizasyon
        waveform = waveform / (np.max(np.abs(waveform)) + 1e-8)

        # Minimum uzunluk kontrolü
        min_samples = int(0.96 * 16000)
        if len(waveform) < min_samples:
            waveform = np.pad(waveform, (0, min_samples - len(waveform)))

        # YAMNet embedding
        _, embeddings, _ = self.yamnet_model(waveform)
        avg_embedding = tf.reduce_mean(embeddings, axis=0).numpy()

        return avg_embedding

    def extract_traditional_features(self, audio_path: str) -> np.ndarray:
        """
        Ses dosyasından traditional features çıkarır

        Args:
            audio_path: Ses dosyası yolu

        Returns:
            np.ndarray: Feature vektörü
        """
        try:
            from src.audio_processing.classification_processor import (
                ClassificationProcessor,
            )
            from src.feature_extraction.classification_extractor import (
                ClassificationExtractor,
            )
        except ImportError as e:
            raise ImportError(f"Feature extraction modülleri yüklenemedi: {e}")

        # 22kHz'de yükle
        audio_processor = ClassificationProcessor(sample_rate=22050)
        feature_extractor = ClassificationExtractor(sample_rate=22050)

        audio_data, _ = audio_processor.load_audio(audio_path)
        processed_audio = audio_processor.preprocess(audio_data)
        features = feature_extractor.extract_features(processed_audio)

        # Features'ın tipini kontrol et ve uygun şekilde dönüştür
        if isinstance(features, dict):
            # Dict -> array
            features_array = np.array(list(features.values()))
        elif isinstance(features, np.ndarray):
            # Zaten array ise düzleştir
            features_array = features.flatten()
        elif isinstance(features, list):
            # List ise array'e çevir
            features_array = np.array(features)
        else:
            raise TypeError(f"Beklenmeyen feature tipi: {type(features)}")

        # Debug: feature boyutunu göster
        print(f"   📏 Feature shape: {features_array.shape}")

        return features_array

    def predict_single_audio(
        self,
        audio_path: str,
        model_path: str,
        model_type: Literal["yamnet", "traditional"],
        verbose: bool = True,
    ) -> Dict[str, any]:
        """
        Tek ses dosyası için tahmin yapar

        Args:
            audio_path: Ses dosyası yolu
            model_path: Eğitilmiş model dizini
            model_type: "yamnet" veya "traditional"
            verbose: Detaylı çıktı

        Returns:
            Dict: Tahmin sonuçları
        """
        audio_path = Path(audio_path)

        if not audio_path.exists():
            raise FileNotFoundError(f"❌ Ses dosyası bulunamadı: {audio_path}")

        if verbose:
            print(f"\n🎵 Tahmin yapılıyor: {audio_path.name}")

        # Feature extraction
        if model_type == "yamnet":
            if verbose:
                print("   📊 YAMNet embedding çıkarılıyor...")
            features = self.extract_yamnet_embedding(str(audio_path))
            feature_names = [f"embedding_{i}" for i in range(len(features))]
        else:
            if verbose:
                print("   📊 Traditional features çıkarılıyor...")
            features = self.extract_traditional_features(str(audio_path))
            feature_names = [f"feature_{i}" for i in range(len(features))]

        # DataFrame oluştur
        df = pd.DataFrame([features], columns=feature_names)

        # Model yükle ve tahmin yap
        if verbose:
            print("   🤖 Model yükleniyor...")
        predictor = TabularPredictor.load(model_path)

        prediction = predictor.predict(df)[0]
        probabilities = predictor.predict_proba(df).iloc[0].to_dict()

        result = {
            "file": audio_path.name,
            "prediction": prediction,
            "probabilities": probabilities,
            "confidence": probabilities[prediction],
        }

        if verbose:
            print(f"   ✅ Tahmin: {prediction} (güven: {result['confidence']:.2%})")
            print(f"   📊 Olasılıklar:")
            for instrument, prob in sorted(
                probabilities.items(), key=lambda x: x[1], reverse=True
            ):
                print(f"      {instrument}: {prob:.2%}")

        return result

    def batch_predict(
        self,
        audio_dir: str,
        model_path: str,
        model_type: Literal["yamnet", "traditional"],
        ground_truth_map: Optional[Dict[str, str]] = None,
    ) -> Dict[str, any]:
        """
        Klasördeki tüm ses dosyaları için toplu tahmin

        Args:
            audio_dir: Ses dosyalarının bulunduğu klasör
            model_path: Eğitilmiş model dizini
            model_type: "yamnet" veya "traditional"
            ground_truth_map: {dosya_adı: gerçek_etiket} (opsiyonel)

        Returns:
            Dict: Toplu tahmin sonuçları
        """
        audio_dir = Path(audio_dir)

        if not audio_dir.exists():
            raise FileNotFoundError(f"❌ Klasör bulunamadı: {audio_dir}")

        # Ses dosyalarını bul
        audio_extensions = {".wav", ".mp3", ".flac", ".m4a", ".aiff"}
        audio_files = [
            f for f in audio_dir.rglob("*") if f.suffix.lower() in audio_extensions
        ]

        if not audio_files:
            print(f"⚠️ {audio_dir} içinde ses dosyası bulunamadı!")
            return {}

        print(f"🎵 {len(audio_files)} ses dosyası bulundu")
        print(f"🤖 Model: {Path(model_path).name}")
        print(f"📊 Tip: {model_type.upper()}")
        print("=" * 70)

        results = []
        y_true = []
        y_pred = []

        for i, audio_file in enumerate(audio_files, 1):
            print(f"\n[{i}/{len(audio_files)}] {audio_file.name}")

            try:
                result = self.predict_single_audio(
                    str(audio_file), model_path, model_type, verbose=False
                )

                results.append(result)
                y_pred.append(result["prediction"])

                # Ground truth varsa ekle
                if ground_truth_map and audio_file.name in ground_truth_map:
                    true_label = ground_truth_map[audio_file.name]
                    y_true.append(true_label)
                    is_correct = true_label == result["prediction"]
                    print(
                        f"   Gerçek: {true_label} | Tahmin: {result['prediction']} | {'✅' if is_correct else '❌'}"
                    )
                else:
                    print(
                        f"   Tahmin: {result['prediction']} (güven: {result['confidence']:.2%})"
                    )

            except Exception as e:
                print(f"   ⚠️ Hata: {e}")

        # Özet istatistikler
        summary = {
            "total_files": len(audio_files),
            "successful_predictions": len(results),
            "failed_predictions": len(audio_files) - len(results),
            "results": results,
        }

        # Eğer ground truth varsa değerlendirme yap
        if y_true and y_pred and len(y_true) == len(y_pred):
            accuracy = accuracy_score(y_true, y_pred)
            conf_matrix = confusion_matrix(y_true, y_pred)
            class_report = classification_report(y_true, y_pred)

            summary["evaluation"] = {
                "accuracy": accuracy,
                "confusion_matrix": conf_matrix.tolist(),
                "classification_report": class_report,
            }

            print("\n" + "=" * 70)
            print("📊 DEĞERLENDİRME SONUÇLARI")
            print("=" * 70)
            print(f"✅ Doğruluk (Accuracy): {accuracy:.2%}")
            print(f"\n📋 Detaylı Rapor:")
            print(class_report)

        # Tahmin dağılımı
        from collections import Counter

        pred_dist = Counter(y_pred)

        print("\n" + "=" * 70)
        print("📈 TAHMİN DAĞILIMI")
        print("=" * 70)
        for instrument, count in pred_dist.most_common():
            print(f"{instrument}: {count} dosya ({count/len(y_pred)*100:.1f}%)")

        return summary


def main():
    """
    Ana program - Menü sistemi ile eğitim veya tahmin
    """
    print("🎼 AUTOML CLASSIFIER - Eğitim & Tahmin Sistemi")
    print("=" * 70)

    # Classifier oluştur
    classifier = AutoMLClassifier()

    print("\n🎯 Ne yapmak istersiniz?")
    print("  1️⃣  Model Eğitimi (Yeni model oluştur)")
    print("  2️⃣  Tahmin (Mevcut model ile tahmin yap)")
    print("  3️⃣  Çıkış")

    mode = input("\nSeçiminiz (1/2/3): ").strip()

    # ==================== EĞİTİM MODU ====================
    if mode == "1":
        print("\n" + "=" * 70)
        print("📚 EĞİTİM MODU")
        print("=" * 70)

        print("\n🎯 Hangi dataset tipini kullanmak istersiniz?")
        print("  1️⃣  YAMNet Embeddings (16kHz, 1024-dim) - ÖNERİLEN")
        print("  2️⃣  Traditional Features (22kHz, handcrafted)")

        choice = input("\nSeçiminiz (1/2): ").strip()

        dataset_type = "yamnet" if choice == "1" else "traditional"

        print(f"\n✅ {dataset_type.upper()} seçildi")

        # Kullanıcıya zaman/kalite seçeneği sun
        print("\n⏱️ Eğitim süresi ve kalite seçenekleri:")
        print("  1️⃣  Hızlı Deneme (10 dk) - ~%88-90 accuracy")
        print("  2️⃣  Normal Kalite (30 dk) - ~%95-95.5 accuracy [ÖNERİLEN]")
        print("  3️⃣  Yüksek Kalite (60 dk) - ~%95.5-96 accuracy")
        print("  4️⃣  En İyi Kalite (2 saat) - ~%96-96.5 accuracy")

        time_choice = (
            input("\nSüre seçiminiz (1/2/3/4) [varsayılan: 2]: ").strip() or "2"
        )

        time_configs = {
            "1": {
                "time_limit": 600,
                "presets": "optimize_for_deployment",
                "name": "Hızlı",
            },
            "2": {"time_limit": 1800, "presets": "medium_quality", "name": "Normal"},
            "3": {"time_limit": 3600, "presets": "high_quality", "name": "Yüksek"},
            "4": {"time_limit": 7200, "presets": "best_quality", "name": "En İyi"},
        }

        config = time_configs.get(time_choice, time_configs["2"])

        print(f"\n✅ {config['name']} kalite seçildi")
        print(f"\n⚙️ Training parametreleri:")
        print(
            f"  Time limit: {config['time_limit']} saniye ({config['time_limit']/60:.0f} dakika)"
        )
        print(f"  Preset: {config['presets']}")
        print(f"  Eval metric: accuracy")

        confirm = input("\nDevam etmek istiyor musunuz? (y/n): ").strip().lower()

        if confirm != "y":
            print("❌ İşlem iptal edildi")
            return

        # Tam pipeline'ı çalıştır
        result = classifier.full_pipeline(
            dataset_type=dataset_type,
            time_limit=config["time_limit"],
            presets=config["presets"],
        )

        print("\n🎉 Eğitim başarıyla tamamlandı!")
        print(f"   📁 Model yolu: {result['model_path']}")
        print(f"   📊 Test accuracy: {result['test_accuracy']:.4f}")
        print(f"   📊 Test F1-score: {result['test_f1']:.4f}")

    # ==================== TAHMİN MODU ====================
    elif mode == "2":
        print("\n" + "=" * 70)
        print("🔮 TAHMİN MODU")
        print("=" * 70)

        # Mevcut modelleri listele
        print("\n📂 Mevcut eğitilmiş modeller:")
        models = classifier.list_available_models()

        if not models:
            print("❌ Hiç eğitilmiş model bulunamadı!")
            print("💡 Önce model eğitimi yapmalısınız (Seçenek 1)")
            return

        for i, model in enumerate(models, 1):
            print(f"  {i}️⃣  {model['name']} ({model['type']})")

        # Model seç
        model_choice = input(
            f"\nHangi modeli kullanmak istersiniz? (1-{len(models)}): "
        ).strip()

        try:
            model_idx = int(model_choice) - 1
            selected_model = models[model_idx]
            print(f"✅ Seçilen model: {selected_model['name']}")
        except (ValueError, IndexError):
            print("❌ Geçersiz seçim!")
            return

        # Tahmin tipi seç
        print("\n🎵 Tahmin yapmak istediğiniz seçeneği seçin:")
        print("  1️⃣  Tek ses dosyası")
        print("  2️⃣  Klasördeki tüm ses dosyaları")

        pred_type = input("\nSeçiminiz (1/2): ").strip()

        if pred_type == "1":
            # Tek dosya tahmini
            audio_path = input("\n📁 Ses dosyası yolunu girin: ").strip()

            if not Path(audio_path).exists():
                print(f"❌ Dosya bulunamadı: {audio_path}")
                return

            try:
                result = classifier.predict_single_audio(
                    audio_path=audio_path,
                    model_path=selected_model["path"],
                    model_type=selected_model["type"],
                    verbose=True,
                )

                print("\n" + "=" * 70)
                print("✅ TAHMİN TAMAMLANDI")
                print("=" * 70)
                print(f"📁 Dosya: {result['file']}")
                print(f"🎵 Tahmin: {result['prediction']}")
                print(f"💯 Güven: {result['confidence']:.2%}")

            except Exception as e:
                print(f"❌ Tahmin hatası: {e}")

        elif pred_type == "2":
            # Toplu tahmin
            audio_dir = input(
                "\n📁 Ses dosyalarının bulunduğu klasör yolunu girin: "
            ).strip()

            if not Path(audio_dir).exists():
                print(f"❌ Klasör bulunamadı: {audio_dir}")
                return

            # Ground truth sorgusu
            use_gt = (
                input("\n❓ Ground truth (gerçek etiketler) var mı? (y/n): ")
                .strip()
                .lower()
            )

            ground_truth_map = None
            if use_gt == "y":
                print("\n📋 Ground truth formatı:")
                print("   Klasör yapısı: audio_dir/enstruman_adı/dosyalar.wav")
                print("   Veya manuel etiket dosyası: labels.txt (dosya_adı:etiket)")

                gt_method = input(
                    "\nYapı: 1=Klasör yapısı, 2=Etiket dosyası (1/2): "
                ).strip()

                if gt_method == "1":
                    # Klasör yapısından otomatik
                    ground_truth_map = {}
                    audio_dir_path = Path(audio_dir)
                    for audio_file in audio_dir_path.rglob("*.wav"):
                        # Eğer dosya bir alt klasördeyse, klasör adı = etiket
                        if audio_file.parent != audio_dir_path:
                            ground_truth_map[audio_file.name] = audio_file.parent.name
                    print(
                        f"✅ {len(ground_truth_map)} dosya için ground truth tespit edildi"
                    )

                elif gt_method == "2":
                    label_file = input("Etiket dosyası yolu: ").strip()
                    if Path(label_file).exists():
                        ground_truth_map = {}
                        with open(label_file, "r") as f:
                            for line in f:
                                if ":" in line:
                                    filename, label = line.strip().split(":", 1)
                                    ground_truth_map[filename] = label
                        print(f"✅ {len(ground_truth_map)} etiket yüklendi")

            try:
                summary = classifier.batch_predict(
                    audio_dir=audio_dir,
                    model_path=selected_model["path"],
                    model_type=selected_model["type"],
                    ground_truth_map=ground_truth_map,
                )

                print("\n" + "=" * 70)
                print("✅ TOPLU TAHMİN TAMAMLANDI")
                print("=" * 70)
                print(f"📊 Toplam dosya: {summary['total_files']}")
                print(f"✅ Başarılı: {summary['successful_predictions']}")
                print(f"❌ Başarısız: {summary['failed_predictions']}")

                if "evaluation" in summary:
                    print(f"\n📈 Model Performansı:")
                    print(f"   Accuracy: {summary['evaluation']['accuracy']:.2%}")

            except Exception as e:
                print(f"❌ Toplu tahmin hatası: {e}")

        else:
            print("❌ Geçersiz seçim!")

    # ==================== ÇIKIŞ ====================
    elif mode == "3":
        print("👋 Çıkış yapılıyor...")
        return

    else:
        print("❌ Geçersiz seçim!")


if __name__ == "__main__":
    main()
