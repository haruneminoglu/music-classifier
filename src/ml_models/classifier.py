# src/ml_models/classifier.py

import sys
import os
from pathlib import Path

# Path düzeltmesi
current_dir = os.path.dirname(os.path.abspath(__file__))  # ml_models/
src_dir = os.path.dirname(current_dir)  # src/
project_root = os.path.dirname(src_dir)  # proje kökü
sys.path.insert(0, project_root)

import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
import pickle
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import json
from datetime import datetime
import warnings
from tqdm import tqdm

# Kendi modüllerden import
try:
    from src.audio_processing.classification_processor import ClassificationProcessor
    from src.feature_extraction.classification_extractor import ClassificationExtractor
    from src.data_managers.instrument_data_manager import DataManager

    print("✅ Proje modülleri başarıyla import edildi")
except ImportError as e:
    print(f"⚠️ Import uyarısı: {e}")
    print("Lütfen proje dizin yapısını kontrol edin")


class InstrumentClassifier:
    """
    YAMNet tabanlı enstrüman sınıflandırma modeli
    Good Sounds veri seti için optimize edilmiş
    DataManager ile uyumlu çalışır
    """

    def __init__(
        self,
        instruments: List[str] = ["cello", "clarinet", "flute", "trumpet", "violin"],
        sample_rate: int = 16000,  # YAMNet için gerekli
        model_save_dir: str = "models/instrument_classifier",
    ):
        """
        InstrumentClassifier başlatıcısı

        Args:
            instruments: Hedef enstrüman listesi
            sample_rate: YAMNet için 16kHz gerekli
            model_save_dir: Model kayıt dizini
        """
        self.instruments = instruments
        self.num_classes = len(instruments)
        self.sample_rate = sample_rate
        self.model_save_dir = Path(model_save_dir)
        self.model_save_dir.mkdir(parents=True, exist_ok=True)

        # Label encoder
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(instruments)

        # Audio processor (16kHz için)
        self.audio_processor = ClassificationProcessor(sample_rate=sample_rate)

        # Model components
        self.yamnet_model = None
        self.classification_head = None
        self.full_model = None

        # Training history
        self.training_history = None
        self.evaluation_results = {}

        print(f"🎵 InstrumentClassifier hazırlandı")
        print(f"  Enstrümanlar: {', '.join(instruments)}")
        print(f"  Sample Rate: {sample_rate} Hz (YAMNet uyumlu)")
        print(f"  Model dizini: {model_save_dir}")

    def load_yamnet(self):
        """YAMNet pre-trained modelini yükler"""
        try:
            print("📥 YAMNet modeli yükleniyor...")
            self.yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")
            print("✅ YAMNet başarıyla yüklendi")
            return True
        except Exception as e:
            print(f"❌ YAMNet yükleme hatası: {e}")
            return False

    def create_classification_head(
        self,
        embedding_dim: int = 1024,
        hidden_units: List[int] = [512, 256],
        dropout_rate: float = 0.3,
    ):
        """
        YAMNet embeddings'leri için classification head oluşturur

        Args:
            embedding_dim: YAMNet embedding boyutu (1024)
            hidden_units: Hidden layer boyutları
            dropout_rate: Dropout oranı
        """
        print("🏗️ Classification head oluşturuluyor...")

        inputs = tf.keras.Input(shape=(embedding_dim,), name="yamnet_embeddings")
        x = inputs

        # Hidden layers
        for i, units in enumerate(hidden_units):
            x = tf.keras.layers.Dense(units, activation="relu", name=f"dense_{i+1}")(x)
            x = tf.keras.layers.BatchNormalization(name=f"bn_{i+1}")(x)
            x = tf.keras.layers.Dropout(dropout_rate, name=f"dropout_{i+1}")(x)

        # Output layer
        outputs = tf.keras.layers.Dense(
            self.num_classes, activation="softmax", name="instrument_predictions"
        )(x)

        self.classification_head = tf.keras.Model(
            inputs=inputs, outputs=outputs, name="InstrumentClassificationHead"
        )

        print(f"✅ Classification head oluşturuldu")
        print(f"  Input: {embedding_dim}D embeddings")
        print(f"  Hidden layers: {hidden_units}")
        print(f"  Output: {self.num_classes} classes")

        return self.classification_head

    def create_full_model(self):
        """YAMNet + Classification Head'den tam model oluşturur"""
        if self.yamnet_model is None:
            if not self.load_yamnet():
                raise ValueError("YAMNet yüklenemedi")

        if self.classification_head is None:
            self.create_classification_head()

        # Full model wrapper
        @tf.function
        def model_fn(waveform):
            # YAMNet embeddings çıkar
            _, embeddings, _ = self.yamnet_model(waveform)
            # Frame-level embeddings'leri average'la
            averaged_embeddings = tf.reduce_mean(embeddings, axis=0, keepdims=True)
            # Classification
            predictions = self.classification_head(averaged_embeddings)
            return predictions

        self.full_model = model_fn
        print("✅ Full model (YAMNet + Classification Head) hazır")

    def preprocess_audio_for_yamnet(self, audio_path: str) -> np.ndarray:
        """
        Ses dosyasını YAMNet için hazırlar

        Args:
            audio_path: Ses dosyası yolu

        Returns:
            np.ndarray: YAMNet için hazırlanmış waveform
        """
        try:
            # 16kHz'de yükle (YAMNet requirement)
            waveform, _ = librosa.load(audio_path, sr=self.sample_rate, mono=True)

            # Normalizasyon
            waveform = waveform / np.max(np.abs(waveform) + 1e-8)

            # YAMNet minimum 0.96 saniye bekler
            min_samples = int(0.96 * self.sample_rate)
            if len(waveform) < min_samples:
                # Pad if too short
                waveform = np.pad(waveform, (0, min_samples - len(waveform)))

            return waveform.astype(np.float32)

        except Exception as e:
            print(f"⚠️ Audio preprocessing hatası: {e}")
            return None

    def extract_yamnet_embeddings(
        self, audio_files: List[str], labels: List[str], batch_size: int = 16
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Ses dosyalarından YAMNet embeddings çıkarır
        NOT: Bu metod sadece embeddings dataset yoksa kullanılır

        Args:
            audio_files: Ses dosyası yolları
            labels: Karşılık gelen etiketler
            batch_size: Batch boyutu

        Returns:
            Tuple: (embeddings, encoded_labels)
        """
        print(f"🎵 {len(audio_files)} dosyadan YAMNet embeddings çıkarılıyor...")
        print(
            "⚠️  NOT: DataManager ile önceden oluşturulmuş embeddings kullanmak daha hızlıdır!"
        )

        if self.yamnet_model is None:
            if not self.load_yamnet():
                raise ValueError("YAMNet yüklenemedi")

        all_embeddings = []
        valid_labels = []

        with tqdm(total=len(audio_files), desc="Embedding extraction") as pbar:
            for i in range(0, len(audio_files), batch_size):
                batch_files = audio_files[i : i + batch_size]
                batch_labels = labels[i : i + batch_size]

                for file_path, label in zip(batch_files, batch_labels):
                    waveform = self.preprocess_audio_for_yamnet(file_path)

                    if waveform is not None:
                        try:
                            # YAMNet ile embedding çıkar
                            _, embeddings, _ = self.yamnet_model(waveform)
                            # Frame-level embeddings'leri ortala
                            avg_embedding = tf.reduce_mean(embeddings, axis=0)

                            all_embeddings.append(avg_embedding.numpy())
                            valid_labels.append(label)

                        except Exception as e:
                            print(
                                f"⚠️ Embedding çıkarma hatası ({Path(file_path).name}): {e}"
                            )

                    pbar.update(1)

        if not all_embeddings:
            raise ValueError("Hiç embedding çıkarılamadı!")

        embeddings_array = np.array(all_embeddings)
        encoded_labels = self.label_encoder.transform(valid_labels)

        print(f"✅ {len(all_embeddings)} embedding çıkarıldı")

        return embeddings_array, encoded_labels

    def prepare_dataset_from_data_manager(
        self, dataset_path: str
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        DataManager'dan oluşturulmuş dataset'i yükler (OPTIMIZE EDİLDİ)

        Bu metod artık:
        1. Önce dataset'te hazır YAMNet embeddings var mı kontrol eder
        2. Varsa direkt kullanır (HIZLI)
        3. Yoksa file path'lerden çıkarır (YAVAŞ ama backward compatible)

        Args:
            dataset_path: Dataset pickle dosya yolu

        Returns:
            Tuple: (embeddings, labels, file_info)
        """
        print(f"📂 Dataset yükleniyor: {dataset_path}")

        with open(dataset_path, "rb") as f:
            dataset = pickle.load(f)

        # Dataset tipini kontrol et
        dataset_type = dataset.get("embedding_type", "unknown")

        # DURUM 1: Dataset'te hazır YAMNet embeddings var (HIZLI - ÖNERİLEN)
        if "embeddings" in dataset and dataset_type == "yamnet":
            print("✅ Hazır YAMNet embeddings bulundu - direkt yükleniyor (HIZLI)")

            embeddings = dataset["embeddings"]
            labels = dataset["labels"]
            file_info = dataset.get("file_info", [])

            # Numpy array'e çevir (eğer değilse)
            if not isinstance(embeddings, np.ndarray):
                embeddings = np.array(embeddings)

            # Label encoding
            encoded_labels = self.label_encoder.transform(labels)

            print(f"  Embeddings shape: {embeddings.shape}")
            print(f"  Embedding dim: {embeddings.shape[1]}")
            print(f"  Sample count: {len(embeddings)}")

            return embeddings, encoded_labels, file_info

        # DURUM 2: Traditional features var (YAMNet için uygun değil)
        elif "features" in dataset:
            print("⚠️ Traditional features bulundu - YAMNet için uygun değil!")
            print(
                "⚠️ DataManager.create_yamnet_dataset() ile YAMNet embeddings oluşturun"
            )
            raise ValueError(
                "YAMNet classifier için YAMNet embeddings dataset'i gerekli!"
            )

        # DURUM 3: File paths var, embeddings yok (YAVAŞ - backward compatibility)
        elif "file_info" in dataset:
            print(
                "⚠️ Hazır embeddings bulunamadı - file paths'lerden çıkarılacak (YAVAŞ)"
            )
            print(
                "💡 Önerilen: DataManager.create_yamnet_dataset() ile önceden oluşturun"
            )

            file_paths = [info["file_path"] for info in dataset["file_info"]]
            labels = dataset["labels"]

            # YAMNet embeddings çıkar (yavaş işlem)
            embeddings, encoded_labels = self.extract_yamnet_embeddings(
                file_paths, labels
            )

            return embeddings, encoded_labels, dataset["file_info"]

        # DURUM 4: Hiçbir kullanılabilir veri yok
        else:
            print("❌ Dataset'te kullanılabilir veri bulunamadı!")
            print("Dataset içeriği:", list(dataset.keys()))
            raise ValueError("Geçersiz dataset formatı!")

    def train_model(
        self,
        train_dataset_path: str,
        val_dataset_path: str,
        epochs: int = 50,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        use_class_weights: bool = True,
    ):
        """
        Model eğitimi gerçekleştirir

        Args:
            train_dataset_path: Training dataset path
            val_dataset_path: Validation dataset path
            epochs: Epoch sayısı
            batch_size: Batch boyutu
            learning_rate: Learning rate
            use_class_weights: Class imbalance için weight kullan
        """
        print(f"🚀 Model eğitimi başlıyor...")
        print(f"  Epochs: {epochs}")
        print(f"  Batch size: {batch_size}")
        print(f"  Learning rate: {learning_rate}")

        # Dataset'leri hazırla
        print("📂 Training dataset hazırlanıyor...")
        X_train, y_train, train_files = self.prepare_dataset_from_data_manager(
            train_dataset_path
        )

        print("📂 Validation dataset hazırlanıyor...")
        X_val, y_val, val_files = self.prepare_dataset_from_data_manager(
            val_dataset_path
        )

        # Classification head oluştur
        if self.classification_head is None:
            self.create_classification_head()

        # Class weights hesapla
        if use_class_weights:
            class_weights = compute_class_weight(
                "balanced", classes=np.unique(y_train), y=y_train
            )
            class_weight_dict = dict(enumerate(class_weights))
            print(f"📊 Class weights: {class_weight_dict}")
        else:
            class_weight_dict = None

        # Model compile
        self.classification_head.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        # Callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=10, restore_best_weights=True, verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=1
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=str(self.model_save_dir / "best_classification_head.keras"),
                monitor="val_accuracy",
                save_best_only=True,
                verbose=1,
            ),
        ]

        # Training
        print("🎯 Eğitim başlatılıyor...")
        self.training_history = self.classification_head.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            class_weight=class_weight_dict,
            callbacks=callbacks,
            verbose=1,
        )

        # En iyi model'i yükle
        self.classification_head.load_weights(
            str(self.model_save_dir / "best_classification_head.keras")
        )

        # Full model'i oluştur
        self.create_full_model()

        print("✅ Model eğitimi tamamlandı!")

        # Training history kaydet
        self.save_training_history()

        return self.training_history

    def predict_single_file(
        self, audio_path: str, segment_length: float = 3.0, overlap: float = 1.5
    ) -> Dict[str, Any]:
        """
        Tek ses dosyası için prediction yapar

        Args:
            audio_path: Ses dosyası yolu
            segment_length: Segment uzunluğu (saniye)
            overlap: Segment overlap (saniye)

        Returns:
            Dict: Prediction sonuçları
        """
        if self.full_model is None:
            if not self.load_model():
                raise ValueError("Model yüklenmedi!")

        try:
            # Ses dosyasını yükle
            waveform, _ = librosa.load(audio_path, sr=self.sample_rate, mono=True)
            duration = len(waveform) / self.sample_rate

            print(f"🎵 Analiz ediliyor: {Path(audio_path).name} ({duration:.2f}s)")

            # Segment-based prediction for long files
            if duration > segment_length:
                predictions = self._predict_segments(waveform, segment_length, overlap)
            else:
                predictions = self._predict_single_segment(waveform)

            return predictions

        except Exception as e:
            print(f"❌ Prediction hatası: {e}")
            return {"error": str(e)}

    def _predict_segments(
        self, waveform: np.ndarray, segment_length: float, overlap: float
    ) -> Dict[str, Any]:
        """Uzun ses dosyaları için segment-based prediction"""
        segment_samples = int(segment_length * self.sample_rate)
        overlap_samples = int(overlap * self.sample_rate)
        step_samples = segment_samples - overlap_samples

        all_predictions = []
        segment_times = []

        for start in range(0, len(waveform) - segment_samples + 1, step_samples):
            end = start + segment_samples
            segment = waveform[start:end]

            try:
                prediction = self.full_model(segment)
                all_predictions.append(prediction.numpy()[0])
                segment_times.append((start / self.sample_rate, end / self.sample_rate))
            except Exception as e:
                print(f"⚠️ Segment prediction hatası: {e}")

        if not all_predictions:
            return {"error": "No valid predictions"}

        # Ensemble predictions
        all_predictions = np.array(all_predictions)
        mean_prediction = np.mean(all_predictions, axis=0)

        # Results
        predicted_class_idx = np.argmax(mean_prediction)
        predicted_instrument = self.instruments[predicted_class_idx]
        confidence = mean_prediction[predicted_class_idx]

        return {
            "predicted_instrument": predicted_instrument,
            "confidence": float(confidence),
            "probabilities": {
                instrument: float(prob)
                for instrument, prob in zip(self.instruments, mean_prediction)
            },
            "segment_count": len(all_predictions),
            "segment_predictions": all_predictions.tolist(),
            "segment_times": segment_times,
        }

    def _predict_single_segment(self, waveform: np.ndarray) -> Dict[str, Any]:
        """Tek segment için prediction"""
        try:
            prediction = self.full_model(waveform)
            probabilities = prediction.numpy()[0]

            predicted_class_idx = np.argmax(probabilities)
            predicted_instrument = self.instruments[predicted_class_idx]
            confidence = probabilities[predicted_class_idx]

            return {
                "predicted_instrument": predicted_instrument,
                "confidence": float(confidence),
                "probabilities": {
                    instrument: float(prob)
                    for instrument, prob in zip(self.instruments, probabilities)
                },
            }
        except Exception as e:
            return {"error": str(e)}

    def evaluate_model(self, test_dataset_path: str) -> Dict[str, Any]:
        """
        Test dataset üzerinde model evaluation

        Args:
            test_dataset_path: Test dataset path

        Returns:
            Dict: Evaluation results
        """
        print(f"📊 Model evaluation başlıyor...")

        # Test dataset hazırla
        X_test, y_test, test_files = self.prepare_dataset_from_data_manager(
            test_dataset_path
        )

        # Predictions
        y_pred_proba = self.classification_head.predict(X_test, verbose=1)
        y_pred = np.argmax(y_pred_proba, axis=1)

        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        classification_rep = classification_report(
            y_test, y_pred, target_names=self.instruments, output_dict=True
        )

        confusion_mat = confusion_matrix(y_test, y_pred)

        results = {
            "accuracy": accuracy,
            "classification_report": classification_rep,
            "confusion_matrix": confusion_mat.tolist(),
            "predictions": y_pred.tolist(),
            "probabilities": y_pred_proba.tolist(),
            "true_labels": y_test.tolist(),
            "test_files": test_files,
        }

        self.evaluation_results = results

        # Sonuçları yazdır
        print(f"\n📈 Test Sonuçları:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"\n🎵 Enstrüman bazında performans:")
        for instrument in self.instruments:
            if instrument in classification_rep:
                precision = classification_rep[instrument]["precision"]
                recall = classification_rep[instrument]["recall"]
                f1 = classification_rep[instrument]["f1-score"]
                print(f"  {instrument}: P={precision:.3f}, R={recall:.3f}, F1={f1:.3f}")

        # Visualization
        self.plot_confusion_matrix(confusion_mat)

        # Results kaydet
        self.save_evaluation_results()

        return results

    def plot_confusion_matrix(self, confusion_matrix: np.ndarray):
        """Confusion matrix görselleştirmesi"""
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            confusion_matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=self.instruments,
            yticklabels=self.instruments,
        )
        plt.title("Enstrüman Sınıflandırma - Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()

        # Save plot
        plot_path = self.model_save_dir / "confusion_matrix.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        print(f"📊 Confusion matrix kaydedildi: {plot_path}")

        plt.show()

    def plot_training_history(self):
        """Training history görselleştirmesi"""
        if self.training_history is None:
            print("❌ Training history bulunamadı!")
            return

        history = self.training_history.history
        epochs = range(1, len(history["loss"]) + 1)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # Loss plot
        ax1.plot(epochs, history["loss"], "bo-", label="Training Loss")
        ax1.plot(epochs, history["val_loss"], "ro-", label="Validation Loss")
        ax1.set_title("Model Loss")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.legend()
        ax1.grid(True)

        # Accuracy plot
        ax2.plot(epochs, history["accuracy"], "bo-", label="Training Accuracy")
        ax2.plot(epochs, history["val_accuracy"], "ro-", label="Validation Accuracy")
        ax2.set_title("Model Accuracy")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy")
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()

        # Save plot
        plot_path = self.model_save_dir / "training_history.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        print(f"📈 Training history kaydedildi: {plot_path}")

        plt.show()

    def save_model(self, model_name: str = "yamnet_instrument_classifier"):
        """Model'i kaydet"""
        try:
            # Classification head kaydet
            classification_head_path = self.model_save_dir / f"{model_name}_head.keras"
            self.classification_head.save(classification_head_path)

            # Label encoder kaydet
            label_encoder_path = self.model_save_dir / f"{model_name}_label_encoder.pkl"
            with open(label_encoder_path, "wb") as f:
                pickle.dump(self.label_encoder, f)

            # Model config kaydet
            config = {
                "instruments": self.instruments,
                "num_classes": self.num_classes,
                "sample_rate": self.sample_rate,
                "model_architecture": "YAMNet + Custom Classification Head",
                "saved_date": datetime.now().isoformat(),
            }

            config_path = self.model_save_dir / f"{model_name}_config.json"
            with open(config_path, "w") as f:
                json.dump(config, f, indent=2)

            print(f"✅ Model kaydedildi:")
            print(f"  Classification Head: {classification_head_path}")
            print(f"  Label Encoder: {label_encoder_path}")
            print(f"  Config: {config_path}")

        except Exception as e:
            print(f"❌ Model kayıt hatası: {e}")

    def load_model(self, model_name: str = "yamnet_instrument_classifier") -> bool:
        """Kaydedilmiş model'i yükle"""
        try:
            # YAMNet yükle
            if not self.load_yamnet():
                return False

            # Classification head yükle
            classification_head_path = self.model_save_dir / f"{model_name}_head.keras"
            if classification_head_path.exists():
                self.classification_head = tf.keras.models.load_model(
                    classification_head_path
                )
            else:
                print(f"⚠️ Classification head bulunamadı: {classification_head_path}")
                return False

            # Label encoder yükle
            label_encoder_path = self.model_save_dir / f"{model_name}_label_encoder.pkl"
            if label_encoder_path.exists():
                with open(label_encoder_path, "rb") as f:
                    self.label_encoder = pickle.load(f)
            else:
                print(f"⚠️ Label encoder bulunamadı: {label_encoder_path}")
                return False

            # Full model oluştur
            self.create_full_model()

            print(f"✅ Model başarıyla yüklendi: {model_name}")
            return True

        except Exception as e:
            print(f"❌ Model yükleme hatası: {e}")
            return False

    def save_training_history(self):
        """Training history'yi kaydet"""
        if self.training_history is None:
            return

        history_path = self.model_save_dir / "training_history.json"

        history_dict = {}
        for key, values in self.training_history.history.items():
            history_dict[key] = [float(val) for val in values]

        with open(history_path, "w") as f:
            json.dump(history_dict, f, indent=2)

        print(f"📈 Training history kaydedildi: {history_path}")

    def save_evaluation_results(self):
        """Evaluation results'ları kaydet"""
        if not self.evaluation_results:
            return

        results_path = self.model_save_dir / "evaluation_results.json"

        results_to_save = {
            "accuracy": self.evaluation_results["accuracy"],
            "classification_report": self.evaluation_results["classification_report"],
            "confusion_matrix": self.evaluation_results["confusion_matrix"],
            "evaluation_date": datetime.now().isoformat(),
        }

        with open(results_path, "w") as f:
            json.dump(results_to_save, f, indent=2)

        print(f"📊 Evaluation results kaydedildi: {results_path}")


def demo_optimized_pipeline():
    """
    Optimize edilmiş YAMNet fine-tuning pipeline
    DataManager ile uyumlu çalışır
    """
    print("🎵 OPTIMIZE EDİLMİŞ YAMNet Pipeline Demo")
    print("=" * 60)

    # ADIM 1: DataManager ile YAMNet embeddings oluştur
    print("\n📊 ADIM 1: YAMNet Embeddings Dataset Oluşturma")
    dm = DataManager()

    # YAMNet embeddings dataset oluştur (tek seferlik, sonra tekrar kullanılır)
    dataset_path = dm.create_yamnet_dataset(
        output_name="good_sounds_yamnet.pkl",
        use_augmentation=True,
        augmentation_factor=2,
    )

    if not dataset_path:
        print("❌ Dataset oluşturulamadı!")
        return

    # Dataset'i böl
    print("\n📊 ADIM 2: Dataset Bölme")
    split_paths = dm.split_dataset(
        dataset_path, test_size=0.2, val_size=0.1, create_cv=True
    )

    # ADIM 2: Classifier ile training
    print("\n🚀 ADIM 3: Model Training")
    classifier = InstrumentClassifier(
        instruments=["cello", "clarinet", "flute", "trumpet", "violin"],
        model_save_dir="models/yamnet_good_sounds",
    )

    try:
        # Model eğitimi (hazır embeddings ile HIZLI)
        training_history = classifier.train_model(
            train_dataset_path=split_paths["train"],
            val_dataset_path=split_paths["val"],
            epochs=30,
            batch_size=32,
            learning_rate=0.001,
            use_class_weights=True,
        )

        # Görselleştirme
        classifier.plot_training_history()

        # Model kaydet
        classifier.save_model("yamnet_good_sounds_v1")

        # ADIM 3: Evaluation
        print("\n📊 ADIM 4: Model Evaluation")
        evaluation_results = classifier.evaluate_model(split_paths["test"])

        print("\n✅ Pipeline tamamlandı!")
        print(f"📁 Model: {classifier.model_save_dir}")
        print(f"📈 Test Accuracy: {evaluation_results['accuracy']:.4f}")

    except Exception as e:
        print(f"❌ Pipeline hatası: {e}")
        import traceback

        traceback.print_exc()


def quick_prediction_demo(
    classifier_path: str = "models/yamnet_good_sounds", audio_file_path: str = None
):
    """Kaydedilmiş model ile hızlı prediction"""
    print("🎯 Hızlı Prediction Demo")
    print("=" * 30)

    classifier = InstrumentClassifier(model_save_dir=classifier_path)

    if not classifier.load_model("yamnet_good_sounds_v1"):
        print("❌ Model yüklenemedi!")
        return

    if audio_file_path and Path(audio_file_path).exists():
        print(f"🎵 Analiz ediliyor: {audio_file_path}")

        result = classifier.predict_single_file(audio_file_path)

        if "error" not in result:
            print(f"\n🎼 Sonuç:")
            print(f"  Tahmin: {result['predicted_instrument']}")
            print(f"  Güven: {result['confidence']:.3f}")
            print(f"\n📊 Tüm olasılıklar:")
            for instrument, prob in result["probabilities"].items():
                print(f"    {instrument}: {prob:.3f}")
        else:
            print(f"❌ Hata: {result['error']}")
    else:
        print("⚠️ Geçerli ses dosyası belirtilmedi")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🎵 YAMNet Instrument Classifier")
    print("=" * 60)
    print("\nHangi işlemi yapmak istersiniz?")
    print("1. Model Eğitimi (Full Pipeline)")
    print("2. Ses Dosyası Tahmini")
    print("=" * 60)

    try:
        choice = input("\nSeçiminiz (1 veya 2): ").strip()

        if choice == "1":
            print("\n🚀 Model eğitimi başlatılıyor...\n")
            demo_optimized_pipeline()

        elif choice == "2":
            print("\n🎯 Tahmin modu başlatılıyor...\n")

            # Model yükleme
            classifier = InstrumentClassifier(
                model_save_dir="models/yamnet_good_sounds"
            )

            if not classifier.load_model("yamnet_good_sounds_v1"):
                print("❌ Model yüklenemedi!")
                print("💡 Önce '1' seçeneği ile model eğitimi yapmalısınız.")
            else:
                while True:
                    audio_path = input(
                        "\n📂 Ses dosyası yolu (çıkmak için 'q'): "
                    ).strip()

                    if audio_path.lower() == "q":
                        print("\n👋 Çıkılıyor...")
                        break

                    # Tırnak işaretlerini temizle (Windows'ta kopyala-yapıştır için)
                    audio_path = audio_path.strip('"').strip("'")

                    if not Path(audio_path).exists():
                        print(f"❌ Dosya bulunamadı: {audio_path}")
                        continue

                    print(f"\n🎵 Analiz ediliyor: {Path(audio_path).name}")
                    print("-" * 50)

                    result = classifier.predict_single_file(audio_path)

                    if "error" not in result:
                        print(f"\n🎼 TAHMİN SONUCU:")
                        print(
                            f"  🎺 Enstrüman : {result['predicted_instrument'].upper()}"
                        )
                        print(f"  ✨ Güven     : {result['confidence']:.1%}")

                        if "segment_count" in result:
                            print(
                                f"  📊 Segment   : {result['segment_count']} parça analiz edildi"
                            )

                        print(f"\n📊 TÜM OLASILIKLAR:")
                        sorted_probs = sorted(
                            result["probabilities"].items(),
                            key=lambda x: x[1],
                            reverse=True,
                        )
                        for instrument, prob in sorted_probs:
                            bar = "█" * int(prob * 30)
                            print(f"  {instrument:12s}: {bar:30s} {prob:.1%}")

                        print("-" * 50)
                    else:
                        print(f"❌ Hata: {result['error']}")
        else:
            print("❌ Geçersiz seçim! Lütfen 1 veya 2 girin.")

    except KeyboardInterrupt:
        print("\n\n👋 Program sonlandırıldı.")
    except Exception as e:
        print(f"\n❌ Beklenmeyen hata: {e}")
        import traceback

        traceback.print_exc()

    print("\n" + "=" * 60)
    print("🎵 Program tamamlandı!")
    print("=" * 60 + "\n")
