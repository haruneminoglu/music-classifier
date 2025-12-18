# src/ml_models/traditional_classifier.py

import sys
import os
from pathlib import Path

# Path düzeltmesi
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(src_dir)
sys.path.insert(0, project_root)

import numpy as np
import pandas as pd
import pickle
import json
from datetime import datetime
from typing import List, Dict, Tuple, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Scikit-learn
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, cross_val_score
import joblib
import warnings

warnings.filterwarnings("ignore")


class TraditionalInstrumentClassifier:
    """
    Traditional ML modelleri (Random Forest, SVM, etc.) ile enstrüman sınıflandırma

    DataManager.create_feature_dataset() ile oluşturulan traditional features ile çalışır
    22kHz sample rate, handcrafted features (MFCC, spectral, chroma, vb.)
    """

    def __init__(
        self,
        instruments: List[str] = ["cello", "clarinet", "flute", "trumpet", "violin"],
        model_type: str = "random_forest",
        model_save_dir: str = "models/traditional_classifier",
    ):
        """
        TraditionalInstrumentClassifier başlatıcısı

        Args:
            instruments: Hedef enstrüman listesi
            model_type: 'random_forest', 'svm', 'gradient_boosting', 'knn', 'naive_bayes'
            model_save_dir: Model kayıt dizini
        """
        self.instruments = instruments
        self.num_classes = len(instruments)
        self.model_type = model_type
        self.model_save_dir = Path(model_save_dir)
        self.model_save_dir.mkdir(parents=True, exist_ok=True)

        # Label encoder
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(instruments)

        # Feature scaler
        self.scaler = StandardScaler()

        # Model initialization
        self.model = self._create_model(model_type)

        # Training results
        self.training_results = {}
        self.evaluation_results = {}
        self.cv_scores = None

        print(f"🌲 TraditionalInstrumentClassifier hazır")
        print(f"  Model: {model_type}")
        print(f"  Enstrümanlar: {', '.join(instruments)}")
        print(f"  Model dizini: {model_save_dir}")

    def _create_model(self, model_type: str):
        """Model oluşturur"""

        if model_type == "random_forest":
            return RandomForestClassifier(
                n_estimators=200,
                max_depth=30,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features="sqrt",
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
                verbose=0,
            )

        elif model_type == "gradient_boosting":
            return GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                min_samples_split=5,
                min_samples_leaf=2,
                subsample=0.8,
                random_state=42,
                verbose=0,
            )

        elif model_type == "svm":
            return SVC(
                kernel="rbf",
                C=10.0,
                gamma="scale",
                class_weight="balanced",
                probability=True,
                random_state=42,
                verbose=False,
            )

        elif model_type == "knn":
            return KNeighborsClassifier(
                n_neighbors=7, weights="distance", metric="euclidean", n_jobs=-1
            )

        elif model_type == "naive_bayes":
            return GaussianNB()

        else:
            raise ValueError(f"Desteklenmeyen model tipi: {model_type}")

    def load_dataset_from_data_manager(
        self, dataset_path: str
    ) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
        """
        DataManager'dan oluşturulmuş traditional features dataset'ini yükler

        Args:
            dataset_path: Dataset pickle dosya yolu

        Returns:
            Tuple: (features, labels, file_info)
        """
        print(f"📂 Dataset yükleniyor: {dataset_path}")

        with open(dataset_path, "rb") as f:
            dataset = pickle.load(f)

        # ✅ TİP KONTROLÜ EKLE
        if "embeddings" in dataset:
            dataset_type = dataset.get("embedding_type", "unknown")
            raise ValueError(
                f"❌ YANLIŞ DATASET TİPİ!\n"
                f"Bu dataset '{dataset_type}' embeddings içeriyor.\n"
                f"Traditional ML classifier için 'features' dataset'i gerekli.\n\n"
                f"ÇÖZÜM:\n"
                f"  from src.data_managers.instrument_data_manager import DataManager\n"
                f"  dm = DataManager()\n"
                f"  features_path = dm.create_feature_dataset()\n"
                f"  splits = dm.split_dataset(features_path)\n"
            )

        # Dataset tipini kontrol et
        if "features" not in dataset:
            raise ValueError(
                "❌ Dataset'te 'features' bulunamadı!\n"
                "Bu dataset traditional features için değil.\n"
                "DataManager.create_feature_dataset() kullanın."
            )

        features = dataset["features"]
        labels = dataset["labels"]
        file_info = dataset.get("file_info", [])

        # Features'ı numpy array'e çevir
        if not isinstance(features, np.ndarray):
            # Liste içinde numpy array'ler varsa flatten et
            if isinstance(features[0], np.ndarray):
                features = np.array([f.flatten() for f in features])
            else:
                features = np.array(features)

        # Eğer features 2D değilse (örn: her sample bir array), düzleştir
        if len(features.shape) > 2:
            features = features.reshape(features.shape[0], -1)

        # Label encoding
        encoded_labels = self.label_encoder.transform(labels)

        print(f"✅ Dataset yüklendi:")
        print(f"  Features shape: {features.shape}")
        print(f"  Sample count: {len(features)}")
        print(f"  Feature dimension: {features.shape[1]}")
        print(f"  Classes: {len(np.unique(encoded_labels))}")

        return features, encoded_labels, file_info

    def train_model(
        self,
        train_dataset_path: str,
        val_dataset_path: str,
        use_cross_validation: bool = False,
        cv_folds: int = 5,
    ) -> Dict[str, Any]:
        """
        Model eğitimi gerçekleştirir

        Args:
            train_dataset_path: Training dataset path
            val_dataset_path: Validation dataset path
            use_cross_validation: Cross-validation yapılsın mı
            cv_folds: CV fold sayısı

        Returns:
            Dict: Training sonuçları
        """
        print(f"\n{'='*70}")
        print(f"🚀 {self.model_type.upper()} Model Eğitimi Başlıyor")
        print(f"{'='*70}")

        # Dataset'leri yükle
        print("\n📊 Training dataset hazırlanıyor...")
        X_train, y_train, train_files = self.load_dataset_from_data_manager(
            train_dataset_path
        )

        print("\n📊 Validation dataset hazırlanıyor...")
        X_val, y_val, val_files = self.load_dataset_from_data_manager(val_dataset_path)

        # Feature scaling
        print("\n🔧 Feature scaling...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)

        # Cross-validation (opsiyonel)
        if use_cross_validation:
            print(f"\n🔄 {cv_folds}-fold Cross-Validation...")
            self.cv_scores = cross_val_score(
                self.model,
                X_train_scaled,
                y_train,
                cv=cv_folds,
                scoring="accuracy",
                n_jobs=-1,
            )
            print(
                f"  CV Accuracy: {self.cv_scores.mean():.4f} (+/- {self.cv_scores.std():.4f})"
            )

        # Model eğitimi
        print(f"\n🎯 {self.model_type} model eğitiliyor...")
        print(f"  Training samples: {len(X_train_scaled)}")
        print(f"  Validation samples: {len(X_val_scaled)}")

        self.model.fit(X_train_scaled, y_train)

        # Training accuracy
        train_accuracy = self.model.score(X_train_scaled, y_train)
        print(f"  ✅ Training Accuracy: {train_accuracy:.4f}")

        # Validation accuracy
        val_accuracy = self.model.score(X_val_scaled, y_val)
        print(f"  ✅ Validation Accuracy: {val_accuracy:.4f}")

        # Validation predictions için detaylı analiz
        y_val_pred = self.model.predict(X_val_scaled)
        val_report = classification_report(
            y_val, y_val_pred, target_names=self.instruments, output_dict=True
        )

        # Training sonuçları
        self.training_results = {
            "model_type": self.model_type,
            "train_accuracy": train_accuracy,
            "val_accuracy": val_accuracy,
            "cv_scores": (
                self.cv_scores.tolist() if self.cv_scores is not None else None
            ),
            "cv_mean": (
                float(self.cv_scores.mean()) if self.cv_scores is not None else None
            ),
            "cv_std": (
                float(self.cv_scores.std()) if self.cv_scores is not None else None
            ),
            "validation_report": val_report,
            "train_samples": len(X_train),
            "val_samples": len(X_val),
            "feature_dim": X_train.shape[1],
            "training_date": datetime.now().isoformat(),
        }

        # Feature importance (varsa)
        if hasattr(self.model, "feature_importances_"):
            self.training_results["feature_importances"] = (
                self.model.feature_importances_.tolist()
            )
            print(f"  📊 Feature importances hesaplandı")

        print(f"\n✅ Model eğitimi tamamlandı!")
        print(f"{'='*70}")

        return self.training_results

    def evaluate_model(
        self, test_dataset_path: str, plot_results: bool = True
    ) -> Dict[str, Any]:
        """
        Test dataset üzerinde model evaluation

        Args:
            test_dataset_path: Test dataset path
            plot_results: Sonuçları görselleştir

        Returns:
            Dict: Evaluation sonuçları
        """
        print(f"\n{'='*70}")
        print(f"📊 Model Evaluation - Test Dataset")
        print(f"{'='*70}")

        # Test dataset yükle
        X_test, y_test, test_files = self.load_dataset_from_data_manager(
            test_dataset_path
        )

        # Feature scaling
        X_test_scaled = self.scaler.transform(X_test)

        # Predictions
        print("\n🎯 Predictions yapılıyor...")
        y_pred = self.model.predict(X_test_scaled)

        # Probabilities (eğer model destekliyorsa)
        if hasattr(self.model, "predict_proba"):
            y_pred_proba = self.model.predict_proba(X_test_scaled)
        else:
            y_pred_proba = None

        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, support = precision_recall_fscore_support(
            y_test, y_pred, average="weighted"
        )

        # Classification report
        class_report = classification_report(
            y_test, y_pred, target_names=self.instruments, output_dict=True
        )

        # Confusion matrix
        conf_matrix = confusion_matrix(y_test, y_pred)

        # Per-class accuracy
        per_class_accuracy = conf_matrix.diagonal() / conf_matrix.sum(axis=1)

        # Evaluation results
        self.evaluation_results = {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "classification_report": class_report,
            "confusion_matrix": conf_matrix.tolist(),
            "per_class_accuracy": {
                instrument: float(acc)
                for instrument, acc in zip(self.instruments, per_class_accuracy)
            },
            "predictions": y_pred.tolist(),
            "true_labels": y_test.tolist(),
            "test_samples": len(y_test),
            "evaluation_date": datetime.now().isoformat(),
        }

        if y_pred_proba is not None:
            self.evaluation_results["probabilities"] = y_pred_proba.tolist()

        # Sonuçları yazdır
        print(f"\n📈 Test Sonuçları:")
        print(f"  Overall Accuracy: {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1-Score: {f1:.4f}")

        print(f"\n🎵 Enstrüman Bazında Performans:")
        for instrument in self.instruments:
            if instrument in class_report:
                p = class_report[instrument]["precision"]
                r = class_report[instrument]["recall"]
                f = class_report[instrument]["f1-score"]
                s = class_report[instrument]["support"]
                print(
                    f"  {instrument:12} - P: {p:.3f}, R: {r:.3f}, F1: {f:.3f} (n={int(s)})"
                )

        # Görselleştirme
        if plot_results:
            self.plot_confusion_matrix(conf_matrix)
            if hasattr(self.model, "feature_importances_"):
                self.plot_feature_importance(top_n=20)

        # Results kaydet
        self.save_evaluation_results()

        print(f"\n{'='*70}")

        return self.evaluation_results

    def predict_single_file(
        self,
        audio_path: str,
        segment_based: bool = True,
        segment_length: float = 3.0,
        overlap: float = 1.5,
    ) -> Dict[str, Any]:
        """
        Tek ses dosyası için prediction

        Args:
            audio_path: Ses dosyası yolu
            segment_based: Segment-based prediction (uzun dosyalar için)
            segment_length: Segment uzunluğu (saniye)
            overlap: Segment overlap (saniye)

        Returns:
            Dict: Prediction sonuçları
        """
        print(f"🎵 Analiz ediliyor: {Path(audio_path).name}")

        try:
            # Modülleri import et
            from src.audio_processing.classification_processor import (
                ClassificationProcessor,
            )
            from src.feature_extraction.classification_extractor import (
                ClassificationExtractor,
            )

            # Processor'ları başlat (22kHz traditional features için)
            audio_processor = ClassificationProcessor(sample_rate=22050)
            feature_extractor = ClassificationExtractor(sample_rate=22050)

            # Ses dosyasını yükle ve işle
            audio_data, metadata = audio_processor.load_audio(audio_path)
            processed_audio = audio_processor.preprocess(audio_data)

            duration = metadata["duration"]

            # Segment-based prediction (uzun dosyalar için)
            if segment_based and duration > segment_length:
                print(f"  📊 Segment-based prediction (duration: {duration:.2f}s)")

                segments = audio_processor.split_audio_segments(
                    processed_audio, segment_length=segment_length, overlap=overlap
                )

                all_predictions = []
                all_probabilities = []

                for i, segment in enumerate(segments):
                    # Feature extraction
                    features = feature_extractor.extract_features(segment)
                    features_scaled = self.scaler.transform(features.reshape(1, -1))

                    # Prediction
                    pred = self.model.predict(features_scaled)[0]
                    all_predictions.append(pred)

                    if hasattr(self.model, "predict_proba"):
                        proba = self.model.predict_proba(features_scaled)[0]
                        all_probabilities.append(proba)

                # Ensemble predictions
                if all_probabilities:
                    mean_proba = np.mean(all_probabilities, axis=0)
                    predicted_class_idx = np.argmax(mean_proba)
                    confidence = mean_proba[predicted_class_idx]
                else:
                    # Voting
                    predicted_class_idx = int(np.bincount(all_predictions).argmax())
                    confidence = np.bincount(all_predictions)[
                        predicted_class_idx
                    ] / len(all_predictions)
                    mean_proba = np.zeros(self.num_classes)
                    mean_proba[predicted_class_idx] = confidence

                predicted_instrument = self.instruments[predicted_class_idx]

                result = {
                    "predicted_instrument": predicted_instrument,
                    "confidence": float(confidence),
                    "probabilities": {
                        instrument: float(prob)
                        for instrument, prob in zip(self.instruments, mean_proba)
                    },
                    "segment_count": len(segments),
                    "duration": duration,
                    "method": "segment_based_ensemble",
                }

            else:
                # Single prediction
                print(f"  📊 Single prediction (duration: {duration:.2f}s)")

                features = feature_extractor.extract_features(processed_audio)
                features_scaled = self.scaler.transform(features.reshape(1, -1))

                predicted_class_idx = self.model.predict(features_scaled)[0]
                predicted_instrument = self.instruments[predicted_class_idx]

                if hasattr(self.model, "predict_proba"):
                    proba = self.model.predict_proba(features_scaled)[0]
                    confidence = proba[predicted_class_idx]
                else:
                    proba = np.zeros(self.num_classes)
                    proba[predicted_class_idx] = 1.0
                    confidence = 1.0

                result = {
                    "predicted_instrument": predicted_instrument,
                    "confidence": float(confidence),
                    "probabilities": {
                        instrument: float(prob)
                        for instrument, prob in zip(self.instruments, proba)
                    },
                    "duration": duration,
                    "method": "single_prediction",
                }

            print(
                f"  🎼 Tahmin: {result['predicted_instrument']} ({result['confidence']:.3f})"
            )

            return result

        except Exception as e:
            print(f"❌ Prediction hatası: {e}")
            import traceback

            traceback.print_exc()
            return {"error": str(e)}

    def hyperparameter_tuning(
        self, train_dataset_path: str, param_grid: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Grid search ile hyperparameter tuning

        Args:
            train_dataset_path: Training dataset path
            param_grid: Parameter grid (None ise default kullanılır)

        Returns:
            Dict: En iyi parametreler ve sonuçlar
        """
        print(f"\n{'='*70}")
        print(f"🔍 Hyperparameter Tuning - {self.model_type.upper()}")
        print(f"{'='*70}")

        # Dataset yükle
        X_train, y_train, _ = self.load_dataset_from_data_manager(train_dataset_path)
        X_train_scaled = self.scaler.fit_transform(X_train)

        # Default parameter grids
        if param_grid is None:
            if self.model_type == "random_forest":
                param_grid = {
                    "n_estimators": [100, 200, 300],
                    "max_depth": [20, 30, 40],
                    "min_samples_split": [2, 5, 10],
                    "max_features": ["sqrt", "log2"],
                }
            elif self.model_type == "svm":
                param_grid = {
                    "C": [0.1, 1.0, 10.0],
                    "gamma": ["scale", "auto"],
                    "kernel": ["rbf", "linear"],
                }
            else:
                print("⚠️ Bu model için default param_grid yok, kendi grid'inizi verin")
                return {}

        print(f"\n🔧 Grid Search parametreleri:")
        for param, values in param_grid.items():
            print(f"  {param}: {values}")

        # Grid Search
        grid_search = GridSearchCV(
            estimator=self.model,
            param_grid=param_grid,
            cv=5,
            scoring="accuracy",
            n_jobs=-1,
            verbose=2,
        )

        print(f"\n🚀 Grid Search başlatılıyor...")
        grid_search.fit(X_train_scaled, y_train)

        # Sonuçlar
        results = {
            "best_params": grid_search.best_params_,
            "best_score": float(grid_search.best_score_),
            "cv_results": {
                "mean_scores": grid_search.cv_results_["mean_test_score"].tolist(),
                "std_scores": grid_search.cv_results_["std_test_score"].tolist(),
                "params": [str(p) for p in grid_search.cv_results_["params"]],
            },
        }

        print(f"\n✅ Grid Search tamamlandı!")
        print(f"  En iyi score: {results['best_score']:.4f}")
        print(f"  En iyi parametreler:")
        for param, value in results["best_params"].items():
            print(f"    {param}: {value}")

        # En iyi modeli kullan
        self.model = grid_search.best_estimator_

        print(f"\n{'='*70}")

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
            cbar_kws={"label": "Count"},
        )

        plt.title(
            f"{self.model_type.upper()} - Confusion Matrix",
            fontsize=14,
            fontweight="bold",
        )
        plt.xlabel("Predicted", fontsize=12)
        plt.ylabel("Actual", fontsize=12)
        plt.tight_layout()

        # Save
        plot_path = self.model_save_dir / f"{self.model_type}_confusion_matrix.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        print(f"📊 Confusion matrix kaydedildi: {plot_path}")

        plt.show()

    def plot_feature_importance(self, top_n: int = 20):
        """Feature importance görselleştirmesi (Random Forest için)"""
        if not hasattr(self.model, "feature_importances_"):
            print("⚠️ Bu model feature importance desteklemiyor")
            return

        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1][:top_n]

        plt.figure(figsize=(12, 8))
        plt.bar(range(top_n), importances[indices])
        plt.xticks(range(top_n), [f"F{i}" for i in indices], rotation=45)
        plt.xlabel("Feature Index")
        plt.ylabel("Importance")
        plt.title(f"{self.model_type.upper()} - Top {top_n} Feature Importances")
        plt.tight_layout()

        # Save
        plot_path = self.model_save_dir / f"{self.model_type}_feature_importance.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        print(f"📊 Feature importance kaydedildi: {plot_path}")

        plt.show()

    def save_model(self, model_name: str = None):
        """Model ve ilgili bileşenleri kaydet"""
        if model_name is None:
            model_name = f"{self.model_type}_classifier"

        try:
            # Model kaydet
            model_path = self.model_save_dir / f"{model_name}_model.pkl"
            joblib.dump(self.model, model_path)

            # Scaler kaydet
            scaler_path = self.model_save_dir / f"{model_name}_scaler.pkl"
            joblib.dump(self.scaler, scaler_path)

            # Label encoder kaydet
            encoder_path = self.model_save_dir / f"{model_name}_encoder.pkl"
            joblib.dump(self.label_encoder, encoder_path)

            # Config kaydet
            config = {
                "model_type": self.model_type,
                "instruments": self.instruments,
                "num_classes": self.num_classes,
                "training_results": self.training_results,
                "saved_date": datetime.now().isoformat(),
            }

            config_path = self.model_save_dir / f"{model_name}_config.json"
            with open(config_path, "w") as f:
                json.dump(config, f, indent=2)

            print(f"\n✅ Model kaydedildi:")
            print(f"  Model: {model_path}")
            print(f"  Scaler: {scaler_path}")
            print(f"  Encoder: {encoder_path}")
            print(f"  Config: {config_path}")

        except Exception as e:
            print(f"❌ Model kaydetme hatası: {e}")

    def load_model(self, model_name: str = None) -> bool:
        """Kaydedilmiş model'i yükle"""
        if model_name is None:
            model_name = f"{self.model_type}_classifier"

        try:
            # Model yükle
            model_path = self.model_save_dir / f"{model_name}_model.pkl"
            self.model = joblib.load(model_path)

            # Scaler yükle
            scaler_path = self.model_save_dir / f"{model_name}_scaler.pkl"
            self.scaler = joblib.load(scaler_path)

            # Label encoder yükle
            encoder_path = self.model_save_dir / f"{model_name}_encoder.pkl"
            self.label_encoder = joblib.load(encoder_path)

            # Config yükle
            config_path = self.model_save_dir / f"{model_name}_config.json"
            if config_path.exists():
                with open(config_path, "r") as f:
                    config = json.load(f)
                    self.training_results = config.get("training_results", {})

            print(f"✅ Model başarıyla yüklendi: {model_name}")
            return True

        except Exception as e:
            print(f"❌ Model yükleme hatası: {e}")
            return False

    def save_evaluation_results(self):
        """Evaluation results'ları kaydet"""
        if not self.evaluation_results:
            return

        results_path = (
            self.model_save_dir / f"{self.model_type}_evaluation_results.json"
        )

        with open(results_path, "w") as f:
            json.dump(self.evaluation_results, f, indent=2)

        print(f"📊 Evaluation results kaydedildi: {results_path}")


def demo_traditional_classifier_pipeline():
    """
    Traditional ML classifier pipeline demo
    DataManager ile oluşturulmuş traditional features dataset kullanır
    """
    print("🌲 TRADITIONAL ML CLASSIFIER PIPELINE DEMO")
    print("=" * 70)

    # Dataset paths (DataManager'dan oluşturulmuş olmalı)
    train_path = "data/processed/datasets/good_sounds_features_train_dataset.pkl"
    val_path = "data/processed/datasets/good_sounds_features_val_dataset.pkl"
    test_path = "data/processed/datasets/good_sounds_features_test_dataset.pkl"

    # Dosya kontrolü
    if not Path(train_path).exists():
        print("⚠️ Dataset bulunamadı!")
        print("Önce DataManager.create_feature_dataset() çalıştırın:")
        print("")
        print("from src.data_managers.instrument_data_manager import DataManager")
        print("dm = DataManager()")
        print("features_path = dm.create_feature_dataset()")
        print("dm.split_dataset(features_path)")
        return

    # ADIM 1: Classifier oluştur
    print("\n" + "=" * 70)
    print("ADIM 1: Random Forest Classifier Oluşturma")
    print("=" * 70)

    classifier = TraditionalInstrumentClassifier(
        model_type="random_forest",  # 'svm', 'gradient_boosting' de olabilir
        model_save_dir="models/random_forest_good_sounds",
    )

    # ADIM 2: Model eğitimi
    print("\n" + "=" * 70)
    print("ADIM 2: Model Eğitimi")
    print("=" * 70)

    training_results = classifier.train_model(
        train_dataset_path=train_path,
        val_dataset_path=val_path,
        use_cross_validation=False,
        cv_folds=5,
    )

    # ADIM 3: Model evaluation
    print("\n" + "=" * 70)
    print("ADIM 3: Model Evaluation")
    print("=" * 70)

    evaluation_results = classifier.evaluate_model(
        test_dataset_path=test_path, plot_results=True
    )

    # ADIM 4: Model kaydet
    print("\n" + "=" * 70)
    print("ADIM 4: Model Kaydetme")
    print("=" * 70)

    classifier.save_model("random_forest_good_sounds_v1")

    # Özet
    print("\n" + "=" * 70)
    print("📊 PİPELİNE ÖZETİ")
    print("=" * 70)
    print(f"\n✅ Model: Random Forest")
    print(f"✅ Training Accuracy: {training_results['train_accuracy']:.4f}")
    print(f"✅ Validation Accuracy: {training_results['val_accuracy']:.4f}")
    print(f"✅ Test Accuracy: {evaluation_results['accuracy']:.4f}")
    print(
        f"✅ Cross-Validation: {training_results['cv_mean']:.4f} (+/- {training_results['cv_std']:.4f})"
    )

    print(f"\n📁 Model dizini: {classifier.model_save_dir}")
    print("\n✨ Pipeline tamamlandı!")


def quick_prediction_demo(audio_file: str):
    """Kaydedilmiş model ile hızlı prediction"""
    print("🎯 Hızlı Prediction Demo")
    print("=" * 30)

    classifier = TraditionalInstrumentClassifier(
        model_save_dir="models/random_forest_good_sounds"
    )

    if not classifier.load_model("random_forest_good_sounds_v1"):
        print("❌ Model yüklenemedi!")
        return

    if Path(audio_file).exists():
        result = classifier.predict_single_file(audio_file)

        if "error" not in result:
            print(f"\n🎼 Sonuç:")
            print(f"  Tahmin: {result['predicted_instrument']}")
            print(f"  Güven: {result['confidence']:.3f}")
            print(f"\n📊 Tüm olasılıklar:")
            for instrument, prob in result["probabilities"].items():
                bar = "█" * int(prob * 30)
                print(f"    {instrument:12} {bar} {prob:.3f}")
        else:
            print(f"❌ Hata: {result['error']}")
    else:
        print(f"❌ Dosya bulunamadı: {audio_file}")


if __name__ == "__main__":
    # Demo seçimi
    print("\n🎵 TRADITIONAL INSTRUMENT CLASSIFIER")
    print("=" * 70)
    print("1. Tam Pipeline (eğitim + evaluation)")
    print("2. Sadece prediction (önceden eğitilmiş model)")
    print("=" * 70)

    choice = input("\nSeçiminiz (1/2): ").strip()

    if choice == "1":
        demo_traditional_classifier_pipeline()
    elif choice == "2":
        audio_file = input("Ses dosyası yolu: ").strip()
        quick_prediction_demo(audio_file)
    else:
        print("Geçersiz seçim!")
