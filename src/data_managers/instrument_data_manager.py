# src/data_managers/instrument_data_manager.py

import sys
import os

# Path düzeltmesi - src/data_managers/ dizininden proje köküne erişim
current_dir = os.path.dirname(os.path.abspath(__file__))  # data_managers/
parent_dir = os.path.dirname(current_dir)  # src/
project_root = os.path.dirname(parent_dir)  # proje kökü
sys.path.insert(0, project_root)

print(f"📂 Proje kök dizini: {project_root}")
print(f"📂 Mevcut dizin: {current_dir}")

import librosa
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
from sklearn.model_selection import train_test_split, StratifiedKFold
import json
import shutil
from tqdm import tqdm
import pickle
from collections import Counter
import tensorflow as tf
import tensorflow_hub as hub


class DataManager:
    """
    Good Sounds veri seti için geliştirilmiş enstrüman tanıma veri yönetimi sınıfı
    YAMNet embeddings ve traditional features desteği
    """

    def __init__(self, base_dir: str = "data"):
        """
        DataManager başlatıcısı

        Args:
            base_dir: Ana veri dizini
        """
        self.base_dir = Path(base_dir)
        self.raw_audio_dir = self.base_dir / "raw_audio" / "good_sounds"
        self.processed_dir = self.base_dir / "processed"
        self.features_dir = self.processed_dir / "features"
        self.datasets_dir = self.processed_dir / "datasets"
        self.models_dir = Path("models")

        # Good Sounds veri setindeki hedef enstrümanlar
        self.target_instruments = ["cello", "clarinet", "flute", "trumpet", "violin"]

        # Model training için gerekli config
        self.training_config = {}

        # YAMNet model cache
        self.yamnet_model = None

        self._create_directory_structure()
        print(f"📁 DataManager hazır - Ana dizin: {self.base_dir}")
        print(f"🎵 Good Sounds veri seti için yapılandırıldı")
        print(f"📂 Raw audio dizini: {self.raw_audio_dir}")

    def _create_directory_structure(self):
        """Proje dizin yapısını oluşturur"""
        directories = [
            self.base_dir,
            self.raw_audio_dir,
            self.processed_dir,
            self.features_dir,
            self.datasets_dir,
            self.processed_dir / "cv_splits",
            self.models_dir,
        ]

        # Good Sounds enstrüman dizinleri
        for instrument in self.target_instruments:
            directories.append(self.raw_audio_dir / instrument)

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

        print(f"✅ Good Sounds için dizin yapısı oluşturuldu")

    def load_yamnet_model(self):
        """YAMNet modelini yükler (cache için) - Geliştirilmiş versiyon"""
        if self.yamnet_model is None:
            try:
                print("📥 YAMNet modeli yükleniyor...")

                # 🔧 ÇÖZÜM 1: Cache'i temizle
                import tempfile

                cache_dir = os.path.join(tempfile.gettempdir(), "tfhub_modules")

                try:
                    if os.path.exists(cache_dir):
                        print("🗑️  TensorFlow Hub cache temizleniyor...")
                        shutil.rmtree(cache_dir)
                        print("✅ Cache temizlendi")
                except Exception as e:
                    print(f"⚠️  Cache temizleme uyarısı (devam ediliyor): {e}")

                # 🔧 ÇÖZÜM 2: Alternatif YAMNet URL'leri
                yamnet_urls = [
                    "https://tfhub.dev/google/yamnet/1",  # Orijinal
                    "https://www.kaggle.com/models/google/yamnet/TensorFlow2/yamnet/1",  # Alternatif 1
                ]

                model_loaded = False
                for i, url in enumerate(yamnet_urls):
                    try:
                        print(f"🔄 Deneme {i+1}/{len(yamnet_urls)}: {url}")
                        self.yamnet_model = hub.load(url)
                        model_loaded = True
                        print(f"✅ Model başarıyla yüklendi!")
                        break
                    except Exception as e:
                        print(f"❌ Bu URL başarısız: {str(e)[:100]}")
                        if i < len(yamnet_urls) - 1:
                            print("🔄 Alternatif URL deneniyor...")
                        continue

                if not model_loaded:
                    print("\n" + "=" * 70)
                    print("❌ YAMNet otomatik yükleme başarısız!")
                    print("=" * 70)
                    print("\n🔧 MANUEL ÇÖZÜMLER:")
                    print("\n1️⃣  TensorFlow sürümünüzü kontrol edin:")
                    print(
                        '   python -c "import tensorflow as tf; print(tf.__version__)"'
                    )
                    print("   Önerilen: TensorFlow 2.10 veya üzeri")
                    print("\n2️⃣  TensorFlow ve TensorFlow Hub'ı güncelleyin:")
                    print("   pip install --upgrade tensorflow tensorflow-hub")
                    print("\n3️⃣  Manuel model indirme:")
                    print("   - https://tfhub.dev/google/yamnet/1")
                    print("   - İndirilen modeli yerel yoldan yükleyin")
                    print("\n4️⃣  Alternatif: Traditional Features kullanın")
                    print("   - Menüden '2' seçeneğini seçin")
                    print("=" * 70)
                    return False

                # 🧪 Model testi
                print("🧪 Model testi yapılıyor...")
                test_audio = np.zeros(16000, dtype=np.float32)  # 1 saniye sessizlik
                _, embeddings, _ = self.yamnet_model(test_audio)
                print(f"✅ Test başarılı! Embedding shape: {embeddings.shape}")
                print(f"   Embedding dim: {embeddings.shape[-1]}")

                return True

            except Exception as e:
                print(f"\n❌ YAMNet yükleme hatası: {e}")
                print(
                    "\n💡 ÖNERİ: Traditional Features dataset kullanmayı deneyin (seçenek 2)"
                )
                return False

        return True

    def scan_audio_files(self) -> Dict[str, List[str]]:
        """
        Good Sounds veri setindeki ses dosyalarını tarar ve organize eder

        Returns:
            Dict: Enstrüman -> dosya listesi mapping
        """
        audio_files = {}
        supported_formats = {".wav", ".mp3", ".flac", ".m4a", ".aiff", ".au"}

        print(f"🔍 Good Sounds veri seti taranıyor...")
        print(f"📂 Tarama dizini: {self.raw_audio_dir}")

        for instrument in self.target_instruments:
            instrument_dir = self.raw_audio_dir / instrument
            files = []

            if instrument_dir.exists():
                print(f"  📁 {instrument} dizini kontrol ediliyor: {instrument_dir}")
                for file_path in instrument_dir.rglob("*"):
                    if (
                        file_path.suffix.lower() in supported_formats
                        and file_path.is_file()
                    ):
                        files.append(str(file_path))
            else:
                print(f"  ⚠️ {instrument} dizini bulunamadı: {instrument_dir}")

            audio_files[instrument] = files
            print(f"  {instrument}: {len(files)} dosya")

        total_files = sum(len(files) for files in audio_files.values())
        print(f"📊 Toplam {total_files} ses dosyası bulundu")

        return audio_files

    def validate_audio_files(
        self, audio_files: Dict[str, List[str]]
    ) -> Dict[str, List[str]]:
        """
        Ses dosyalarını validate eder, bozuk dosyaları filtreler

        Args:
            audio_files: Taranmış ses dosyaları

        Returns:
            Dict: Validate edilmiş ses dosyaları
        """
        print("🔍 Good Sounds veri seti validate ediliyor...")

        validated_files = {}
        total_removed = 0

        for instrument, file_list in audio_files.items():
            valid_files = []

            print(f"  🎵 {instrument} enstrümanı validate ediliyor...")

            for file_path in file_list:
                try:
                    # Dosyayı yüklemeyi dene
                    audio, sr = librosa.load(file_path, duration=1.0)

                    # Minimum süre kontrolü (0.5 saniye)
                    if len(audio) / sr >= 0.5:
                        valid_files.append(file_path)
                    else:
                        print(f"    ⚠️ Çok kısa dosya atlandı: {Path(file_path).name}")
                        total_removed += 1

                except Exception as e:
                    print(
                        f"    ⚠️ Bozuk dosya atlandı: {Path(file_path).name} - {str(e)[:50]}..."
                    )
                    total_removed += 1

            validated_files[instrument] = valid_files
            print(
                f"    ✅ {instrument}: {len(valid_files)}/{len(file_list)} geçerli dosya"
            )

        if total_removed > 0:
            print(f"📊 Toplam {total_removed} dosya filtrelendi")
        else:
            print(f"✅ Tüm dosyalar geçerli")

        return validated_files

    def analyze_class_balance(self, labels: List[str]) -> Dict[str, Any]:
        """
        Sınıf dengesi analizi yapar

        Args:
            labels: Etiket listesi

        Returns:
            Dict: Denge analizi sonuçları
        """
        label_counts = Counter(labels)
        min_count = min(label_counts.values())
        max_count = max(label_counts.values())
        imbalance_ratio = max_count / min_count if min_count > 0 else float("inf")

        analysis = {
            "label_counts": dict(label_counts),
            "min_count": min_count,
            "max_count": max_count,
            "imbalance_ratio": imbalance_ratio,
            "needs_balancing": imbalance_ratio > 2.0,
            "recommended_technique": (
                "class_weights" if imbalance_ratio > 2.0 else "none"
            ),
        }

        print(f"📊 Good Sounds veri seti sınıf dengesi analizi:")
        for instrument, count in label_counts.items():
            print(f"  {instrument}: {count} örnek")
        print(f"  Dengesizlik oranı: {imbalance_ratio:.2f}:1")
        if analysis["needs_balancing"]:
            print(f"  🔄 Önerilen dengeleme: {analysis['recommended_technique']}")
        else:
            print(f"  ✅ Veri seti dengeli")

        return analysis

    def augment_audio_data(
        self,
        audio_data: np.ndarray,
        sample_rate: int = 16000,
        augmentation_factor: int = 3,
    ) -> List[np.ndarray]:
        """
        Good Sounds için optimize edilmiş ses verisi artırımı

        Args:
            audio_data: Orijinal ses verisi
            sample_rate: Örnekleme frekansı (16000 veya 22050)
            augmentation_factor: Kaç adet augmented veri üretilecek

        Returns:
            List: Artırılmış ses verileri
        """
        augmented_data = [audio_data]  # Orijinal dahil

        try:
            # ✅ Sample rate'i lambda içinde kullan (closure ile)
            techniques = [
                # Pitch shifting
                lambda x, sr=sample_rate: librosa.effects.pitch_shift(
                    x, sr=sr, n_steps=1
                ),
                lambda x, sr=sample_rate: librosa.effects.pitch_shift(
                    x, sr=sr, n_steps=-1
                ),
                # Time stretching
                lambda x, sr=sample_rate: librosa.effects.time_stretch(x, rate=0.95),
                lambda x, sr=sample_rate: librosa.effects.time_stretch(x, rate=1.05),
                # Hafif gürültü
                lambda x, sr=sample_rate: x + np.random.normal(0, 0.003, len(x)),
                # Gain variation
                lambda x, sr=sample_rate: x * np.random.uniform(0.8, 1.2),
            ]

            selected_techniques = techniques[
                : min(augmentation_factor, len(techniques))
            ]

            for i, technique in enumerate(selected_techniques):
                try:
                    augmented = technique(audio_data)
                    augmented = np.clip(augmented, -1.0, 1.0)
                    augmented_data.append(augmented)
                except Exception as e:
                    print(f"    ⚠️ Augmentation tekniği {i+1} hatası: {e}")

        except Exception as e:
            print(f"⚠️ Genel augmentation hatası: {e}")

        return augmented_data

    def create_both_datasets(
        self,
        use_augmentation: bool = True,
        yamnet_augmentation_factor: int = 2,
        features_augmentation_factor: int = 3,
    ) -> Dict[str, Optional[str]]:
        """
        Hem YAMNet hem de Traditional Features için dataset oluşturur

        Args:
            use_augmentation: Augmentation kullanılsın mı
            yamnet_augmentation_factor: YAMNet için augmentation faktörü
            features_augmentation_factor: Features için augmentation faktörü

        Returns:
            Dict: Her iki dataset'in dosya yolları
        """
        print("=" * 70)
        print("🎼 GOOD SOUNDS - İKİ DATASET OLUŞTURMA")
        print("=" * 70)

        results = {}

        # 1️⃣ YAMNet Dataset (16kHz)
        print("\n" + "=" * 70)
        print("1️⃣ YAMNet Embeddings Dataset (16kHz)")
        print("=" * 70)

        yamnet_path = self.create_yamnet_dataset(
            output_name="good_sounds_yamnet.pkl",
            use_augmentation=use_augmentation,
            augmentation_factor=yamnet_augmentation_factor,
        )

        if yamnet_path:
            # YAMNet dataset'i böl
            print("\n📊 YAMNet dataset bölünüyor...")
            yamnet_splits = self.split_dataset(
                yamnet_path, test_size=0.2, val_size=0.1, create_cv=False
            )

            results["yamnet"] = {
                "full_dataset": yamnet_path,
                "splits": yamnet_splits,
                "sample_rate": 16000,
                "data_type": "yamnet_embeddings",
            }

            print(f"✅ YAMNet dataset hazır!")
            print(f"   📁 Full: {yamnet_path}")
            print(f"   📁 Train: {yamnet_splits['train']}")
            print(f"   📁 Val: {yamnet_splits['val']}")
            print(f"   📁 Test: {yamnet_splits['test']}")
        else:
            print("❌ YAMNet dataset oluşturulamadı!")
            results["yamnet"] = None

        # 2️⃣ Traditional Features Dataset (22kHz)
        print("\n" + "=" * 70)
        print("2️⃣ Traditional Features Dataset (22kHz)")
        print("=" * 70)

        features_path = self.create_feature_dataset(
            output_name="good_sounds_features.pkl",
            use_augmentation=use_augmentation,
            augmentation_factor=features_augmentation_factor,
        )

        if features_path:
            # Features dataset'i böl
            print("\n📊 Features dataset bölünüyor...")
            features_splits = self.split_dataset(
                features_path, test_size=0.2, val_size=0.1, create_cv=False
            )

            results["features"] = {
                "full_dataset": features_path,
                "splits": features_splits,
                "sample_rate": 22050,
                "data_type": "traditional_features",
            }

            print(f"✅ Traditional Features dataset hazır!")
            print(f"   📁 Full: {features_path}")
            print(f"   📁 Train: {features_splits['train']}")
            print(f"   📁 Val: {features_splits['val']}")
            print(f"   📁 Test: {features_splits['test']}")
        else:
            print("❌ Traditional Features dataset oluşturulamadı!")
            results["features"] = None

        # 📊 Özet
        print("\n" + "=" * 70)
        print("📊 DATASET OLUŞTURMA ÖZETİ")
        print("=" * 70)

        if results.get("yamnet"):
            print(f"\n✅ YAMNet Dataset:")
            print(f"   Sample Rate: 16000 Hz")
            print(f"   Data Type: Embeddings (1024-dim)")
            print(f"   Full Dataset: {results['yamnet']['full_dataset']}")
        else:
            print(f"\n❌ YAMNet Dataset oluşturulamadı")

        if results.get("features"):
            print(f"\n✅ Traditional Features Dataset:")
            print(f"   Sample Rate: 22050 Hz")
            print(f"   Data Type: Handcrafted Features")
            print(f"   Full Dataset: {results['features']['full_dataset']}")
        else:
            print(f"\n❌ Traditional Features Dataset oluşturulamadı")

        print("\n" + "=" * 70)

        return results

    def create_yamnet_dataset(
        self,
        output_name: str = "good_sounds_yamnet.pkl",
        use_augmentation: bool = True,
        augmentation_factor: int = 2,
        batch_size: int = 16,
    ) -> Optional[str]:
        """
        YAMNet embeddings ile dataset oluşturur (ÖNERILEN METOD)

        Args:
            output_name: Çıktı dosya adı
            use_augmentation: Augmentation kullanılsın mı
            augmentation_factor: Her dosya için kaç augmentation
            batch_size: Batch boyutu

        Returns:
            str: Oluşturulan dataset dosya yolu
        """
        print("🎵 YAMNet embeddings dataset oluşturuluyor...")
        print("⚡ Bu metod CNN training için optimize edilmiştir")

        # YAMNet modelini yükle
        if not self.load_yamnet_model():
            print("❌ YAMNet yüklenemedi!")
            return None

        # Dosyaları tara ve validate et
        audio_files = self.scan_audio_files()
        validated_files = self.validate_audio_files(audio_files)

        all_embeddings = []
        all_labels = []
        file_info = []

        total_files = sum(len(files) for files in validated_files.values())

        if total_files == 0:
            print("❌ Geçerli ses dosyası bulunamadı!")
            return None

        print(f"🔄 Augmentation: {'Aktif' if use_augmentation else 'Pasif'}")
        if use_augmentation:
            print(f"  Faktör: x{augmentation_factor}")

        with tqdm(total=total_files, desc="YAMNet embeddings") as pbar:
            for instrument in self.target_instruments:
                file_list = validated_files.get(instrument, [])

                if not file_list:
                    print(f"⚠️ {instrument} için dosya bulunamadı")
                    continue

                for file_path in file_list:
                    try:
                        # 16kHz'de yükle (YAMNet requirement)
                        waveform, _ = librosa.load(file_path, sr=16000, mono=True)

                        # Normalizasyon
                        waveform = waveform / (np.max(np.abs(waveform)) + 1e-8)

                        # Minimum uzunluk kontrolü
                        min_samples = int(0.96 * 16000)
                        if len(waveform) < min_samples:
                            waveform = np.pad(
                                waveform, (0, min_samples - len(waveform))
                            )

                        # Augmentation uygula
                        if use_augmentation:
                            audio_variants = self.augment_audio_data(
                                waveform, 16000, augmentation_factor
                            )
                        else:
                            audio_variants = [waveform]

                        # Her variant için embedding çıkar
                        for i, audio_variant in enumerate(audio_variants):
                            try:
                                # YAMNet ile embedding
                                _, embeddings, _ = self.yamnet_model(audio_variant)
                                # Frame-level embeddings'leri ortala
                                avg_embedding = tf.reduce_mean(
                                    embeddings, axis=0
                                ).numpy()

                                all_embeddings.append(avg_embedding)
                                all_labels.append(instrument)
                                file_info.append(
                                    {
                                        "file_path": file_path,
                                        "instrument": instrument,
                                        "augmentation_id": i,
                                        "is_original": i == 0,
                                        "dataset": "good_sounds",
                                        "embedding_type": "yamnet",
                                    }
                                )

                            except Exception as e:
                                print(
                                    f"⚠️ Embedding hatası ({Path(file_path).name}): {e}"
                                )

                        pbar.set_postfix(
                            {
                                "Instrument": instrument,
                                "Embeddings": len(all_embeddings),
                            }
                        )

                    except Exception as e:
                        print(f"⚠️ Audio load hatası ({Path(file_path).name}): {e}")

                    pbar.update(1)

        if not all_embeddings:
            print("❌ Hiç embedding çıkarılamadı!")
            return None

        # Class balance analizi
        balance_analysis = self.analyze_class_balance(all_labels)

        # Dataset'i oluştur
        dataset = {
            "embeddings": np.array(all_embeddings),  # YAMNet embeddings
            "labels": all_labels,
            "file_info": file_info,
            "instruments": self.target_instruments,
            "embedding_type": "yamnet",
            "embedding_dim": 1024,  # YAMNet embedding size
            "sample_rate": 16000,
            "total_samples": len(all_embeddings),
            "augmentation_used": use_augmentation,
            "augmentation_factor": augmentation_factor if use_augmentation else 0,
            "balance_analysis": balance_analysis,
            "dataset_name": "good_sounds",
            "created_date": pd.Timestamp.now().isoformat(),
        }

        # Dataset'i kaydet
        output_path = self.datasets_dir / output_name

        with open(output_path, "wb") as f:
            pickle.dump(dataset, f)

        # Metadata kaydet
        metadata_path = (
            self.features_dir / f"metadata_{output_name.replace('.pkl', '.json')}"
        )
        metadata = {
            "embedding_type": "yamnet",
            "embedding_dim": 1024,
            "sample_count": len(all_embeddings),
            "instruments": self.target_instruments,
            "class_distribution": balance_analysis["label_counts"],
            "augmentation_info": {
                "used": use_augmentation,
                "factor": augmentation_factor,
            },
            "created_date": pd.Timestamp.now().isoformat(),
        }

        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        # Training config'i güncelle
        self.training_config.update(
            {
                "dataset_path": str(output_path),
                "metadata_path": str(metadata_path),
                "total_samples": len(all_embeddings),
                "embedding_dim": 1024,
                "embedding_type": "yamnet",
                "class_info": balance_analysis,
                "augmentation_info": {
                    "used": use_augmentation,
                    "factor": augmentation_factor,
                },
            }
        )

        print(f"✅ YAMNet dataset oluşturuldu: {output_path}")
        print(f"📊 Toplam örnek: {len(all_embeddings)}")
        print(f"🎼 Enstrüman dağılımı:")

        for instrument, count in balance_analysis["label_counts"].items():
            print(f"  {instrument}: {count} örnek")

        print(f"📁 Metadata: {metadata_path}")

        return str(output_path)

    def create_feature_dataset(
        self,
        output_name: str = "good_sounds_features.pkl",
        use_augmentation: bool = True,
        augmentation_factor: int = 3,
        batch_size: int = 50,
    ) -> Optional[str]:
        """
        Good Sounds veri setinden traditional feature dataset oluşturur
        NOT: YAMNet kullanıyorsanız create_yamnet_dataset() metodunu kullanın

        Args:
            output_name: Çıktı dosya adı
            use_augmentation: Augmentation kullanılsın mı
            augmentation_factor: Her dosya için kaç augmentation
            batch_size: Memory optimization için batch boyutu

        Returns:
            str: Oluşturulan dataset dosya yolu
        """
        print("🎵 Traditional features dataset oluşturuluyor...")
        print("⚠️  YAMNet kullanıyorsanız create_yamnet_dataset() metodunu kullanın")

        # Modülleri import et
        try:
            from src.audio_processing.classification_processor import (
                ClassificationProcessor,
            )
            from src.feature_extraction.classification_extractor import (
                ClassificationExtractor,
            )

            print("✅ Classification processing modülleri başarıyla import edildi")
        except ImportError as e:
            print(f"❌ Import hatası: {e}")
            return None

        # Processor'ları başlat
        audio_processor = ClassificationProcessor(sample_rate=22050)  # 22kHz
        feature_extractor = ClassificationExtractor(sample_rate=22050)  # 22kHz

        # Dosyaları tara ve validate et
        audio_files = self.scan_audio_files()
        validated_files = self.validate_audio_files(audio_files)

        all_features = []
        all_labels = []
        file_info = []

        total_files = sum(len(files) for files in validated_files.values())

        if total_files == 0:
            print("❌ Geçerli ses dosyası bulunamadı!")
            return None

        print(f"🔄 Augmentation: {'Aktif' if use_augmentation else 'Pasif'}")

        batch_features = []
        batch_labels = []
        batch_info = []

        with tqdm(total=total_files, desc="Feature extraction") as pbar:
            for instrument in self.target_instruments:
                file_list = validated_files.get(instrument, [])

                if not file_list:
                    continue

                for file_path in file_list:
                    try:
                        audio_data, metadata = audio_processor.load_audio(file_path)
                        processed_audio = audio_processor.preprocess(audio_data)

                        if use_augmentation:
                            audio_variants = self.augment_audio_data(
                                processed_audio,
                                sample_rate=22050,  # ✅ Sabit değer
                                augmentation_factor=augmentation_factor,
                            )
                        else:
                            audio_variants = [processed_audio]

                        for i, audio_variant in enumerate(audio_variants):
                            features = feature_extractor.extract_features(audio_variant)

                            batch_features.append(features)
                            batch_labels.append(instrument)
                            batch_info.append(
                                {
                                    "file_path": file_path,
                                    "instrument": instrument,
                                    "augmentation_id": i,
                                    "is_original": i == 0,
                                    "dataset": "good_sounds",
                                }
                            )

                        if len(batch_features) >= batch_size:
                            all_features.extend(batch_features)
                            all_labels.extend(batch_labels)
                            file_info.extend(batch_info)

                            batch_features = []
                            batch_labels = []
                            batch_info = []

                    except Exception as e:
                        print(f"⚠️ Hata: {e}")

                    pbar.update(1)

        if batch_features:
            all_features.extend(batch_features)
            all_labels.extend(batch_labels)
            file_info.extend(batch_info)

        if not all_features:
            print("❌ Hiç özellik çıkarılamadı!")
            return None

        balance_analysis = self.analyze_class_balance(all_labels)

        dataset = {
            "features": all_features,
            "labels": all_labels,
            "file_info": file_info,
            "instruments": self.target_instruments,
            "total_samples": len(all_features),
            "feature_type": "traditional",
            "augmentation_used": use_augmentation,
            "augmentation_factor": augmentation_factor if use_augmentation else 0,
            "balance_analysis": balance_analysis,
            "dataset_name": "good_sounds",
            "created_date": pd.Timestamp.now().isoformat(),
        }

        output_path = self.datasets_dir / output_name

        with open(output_path, "wb") as f:
            pickle.dump(dataset, f)

        print(f"✅ Traditional features dataset oluşturuldu: {output_path}")
        print(f"📊 Toplam örnek: {len(all_features)}")

        return str(output_path)

    def split_dataset(
        self,
        dataset_path: str,
        test_size: float = 0.2,
        val_size: float = 0.1,
        create_cv: bool = False,
    ) -> Dict[str, str]:
        """Dataset'i train/val/test olarak böler"""

        print(f"📊 Dataset bölünüyor: test={test_size}, val={val_size}")

        with open(dataset_path, "rb") as f:
            dataset = pickle.load(f)

        # Embedding veya feature'ları al
        if "embeddings" in dataset:
            X = dataset["embeddings"]
            data_key = "embeddings"
        elif "features" in dataset:
            X = dataset["features"]
            data_key = "features"
        else:
            print("❌ Dataset'te embeddings veya features bulunamadı!")
            return {}

        labels = dataset["labels"]
        file_info = dataset["file_info"]

        # Stratified split
        X_temp, X_test, y_temp, y_test, info_temp, info_test = train_test_split(
            X, labels, file_info, test_size=test_size, random_state=42, stratify=labels
        )

        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val, info_train, info_val = train_test_split(
            X_temp,
            y_temp,
            info_temp,
            test_size=val_size_adjusted,
            random_state=42,
            stratify=y_temp,
        )

        # Splits oluştur
        splits = {
            "train": {
                data_key: X_train,
                "labels": y_train,
                "file_info": info_train,
                "split": "train",
                "instruments": self.target_instruments,
                "dataset_name": "good_sounds",
                "embedding_type": dataset.get("embedding_type", "unknown"),  # ✅ EKLE
                "embedding_dim": dataset.get("embedding_dim", None),  # ✅ EKLE
            },
            "val": {
                data_key: X_val,
                "labels": y_val,
                "file_info": info_val,
                "split": "validation",
                "instruments": self.target_instruments,
                "dataset_name": "good_sounds",
                "embedding_type": dataset.get("embedding_type", "unknown"),  # ✅ EKLE
                "embedding_dim": dataset.get("embedding_dim", None),  # ✅ EKLE
            },
            "test": {
                data_key: X_test,
                "labels": y_test,
                "file_info": info_test,
                "split": "test",
                "instruments": self.target_instruments,
                "dataset_name": "good_sounds",
                "embedding_type": dataset.get("embedding_type", "unknown"),  # ✅ EKLE
                "embedding_dim": dataset.get("embedding_dim", None),  # ✅ EKLE
            },
        }

        saved_paths = {}
        for split_name, split_data in splits.items():
            # ✅ DÜZELTİLMİŞ: Orijinal dataset adından tip bilgisini al
            original_filename = Path(
                dataset_path
            ).stem  # "good_sounds_yamnet" veya "good_sounds_features"
            split_path = (
                self.datasets_dir / f"{original_filename}_{split_name}_dataset.pkl"
            )

            with open(split_path, "wb") as f:
                pickle.dump(split_data, f)
            saved_paths[split_name] = str(split_path)

            print(f"  {split_name}: {len(split_data[data_key])} örnek -> {split_path}")

        # Cross-validation splits
        if create_cv:
            cv_paths = self.create_cv_splits(X_train, y_train, info_train, data_key)
            saved_paths["cv_splits"] = cv_paths

        return saved_paths

    def create_cv_splits(
        self, X_train, y_train, info_train, data_key: str, k_folds: int = 5
    ) -> List[str]:
        """Cross-validation splits oluşturur"""
        print(f"🔄 {k_folds}-fold CV splits oluşturuluyor...")

        skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
        cv_paths = []

        cv_dir = self.processed_dir / "cv_splits"
        cv_dir.mkdir(exist_ok=True)

        for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
            fold_data = {
                "fold_number": fold,
                f"train_{data_key}": [X_train[i] for i in train_idx],
                "train_labels": [y_train[i] for i in train_idx],
                "train_info": [info_train[i] for i in train_idx],
                f"val_{data_key}": [X_train[i] for i in val_idx],
                "val_labels": [y_train[i] for i in val_idx],
                "val_info": [info_train[i] for i in val_idx],
                "instruments": self.target_instruments,
                "dataset_name": "good_sounds",
            }

            fold_path = cv_dir / f"good_sounds_fold_{fold}.pkl"
            with open(fold_path, "wb") as f:
                pickle.dump(fold_data, f)

            cv_paths.append(str(fold_path))
            print(f"  Fold {fold}: {len(train_idx)} train, {len(val_idx)} val")

        return cv_paths

    def get_training_config(self) -> Dict[str, Any]:
        """Training konfigürasyonu döndürür"""
        config = self.training_config.copy()
        config.update(
            {
                "instruments": self.target_instruments,
                "num_classes": len(self.target_instruments),
                "dataset_name": "good_sounds",
                "recommendations": {
                    "use_class_balancing": config.get("class_info", {}).get(
                        "needs_balancing", False
                    ),
                    "balancing_method": config.get("class_info", {}).get(
                        "recommended_technique", "class_weights"
                    ),
                    "suggested_epochs": 50,
                    "suggested_batch_size": 32,
                    "suggested_lr": 0.001,
                    "model_type": "yamnet_fine_tuning",
                    "early_stopping_patience": 10,
                },
            }
        )

        return config

    def get_dataset_info(self, dataset_path: str) -> Dict[str, Any]:
        """Dataset bilgilerini döndürür"""
        try:
            with open(dataset_path, "rb") as f:
                dataset = pickle.load(f)

            info = {
                "dataset_name": dataset.get("dataset_name", "unknown"),
                "total_samples": len(dataset.get("labels", [])),
                "instruments": dataset.get("instruments", []),
                "class_distribution": dict(Counter(dataset.get("labels", []))),
                "created_date": dataset.get("created_date", "unknown"),
            }

            if "embeddings" in dataset:
                info["data_type"] = "yamnet_embeddings"
                info["embedding_dim"] = dataset.get("embedding_dim", 1024)
            elif "features" in dataset:
                info["data_type"] = "traditional_features"
                info["feature_dim"] = (
                    len(dataset["features"][0]) if dataset["features"] else 0
                )

            return info

        except Exception as e:
            return {"error": str(e)}

    def list_datasets(self) -> List[Dict[str, Any]]:
        """Mevcut dataset'leri listeler"""
        datasets = []

        for pkl_file in self.datasets_dir.glob("*.pkl"):
            info = self.get_dataset_info(str(pkl_file))
            info["file_path"] = str(pkl_file)
            info["file_name"] = pkl_file.name
            info["file_size_mb"] = pkl_file.stat().st_size / (1024 * 1024)
            datasets.append(info)

        return datasets


def main():
    """Test ve demo fonksiyonu"""
    print("🎼 GOOD SOUNDS DATA MANAGER - Dual Dataset Edition")
    print("=" * 70)

    # DataManager oluştur
    dm = DataManager()

    # Dosyaları kontrol et
    audio_files = dm.scan_audio_files()
    total_files = sum(len(files) for files in audio_files.values())

    if total_files == 0:
        print("⚠️ Ses dosyası bulunamadı!")
        print(f"📁 {dm.raw_audio_dir}/ dizinine ses dosyaları ekleyin")
        return

    print(f"\n✅ {total_files} ses dosyası bulundu!")

    # Kullanıcıya seçenek sun
    print("\n🎯 Dataset oluşturma seçenekleri:")
    print("  1️⃣  Sadece YAMNet (16kHz embeddings)")
    print("  2️⃣  Sadece Traditional Features (22kHz)")
    print("  3️⃣  Her ikisi de (ÖNERİLEN)")

    choice = input("\nSeçiminiz (1/2/3): ").strip()

    if choice == "1":
        # Sadece YAMNet
        print("\n🚀 YAMNet dataset oluşturuluyor...")
        yamnet_path = dm.create_yamnet_dataset(
            output_name="good_sounds_yamnet.pkl",
            use_augmentation=True,
            augmentation_factor=2,
        )

        if yamnet_path:
            splits = dm.split_dataset(
                yamnet_path, test_size=0.2, val_size=0.1, create_cv=False
            )
            print(f"\n✅ YAMNet dataset hazır!")
            print(f"📁 Train: {splits['train']}")
            print(f"📁 Val: {splits['val']}")
            print(f"📁 Test: {splits['test']}")

    elif choice == "2":
        # Sadece Traditional Features
        print("\n🌲 Traditional Features dataset oluşturuluyor...")
        features_path = dm.create_feature_dataset(
            output_name="good_sounds_features.pkl",
            use_augmentation=True,
            augmentation_factor=3,
        )

        if features_path:
            splits = dm.split_dataset(
                features_path, test_size=0.2, val_size=0.1, create_cv=False
            )
            print(f"\n✅ Traditional Features dataset hazır!")
            print(f"📁 Train: {splits['train']}")
            print(f"📁 Val: {splits['val']}")
            print(f"📁 Test: {splits['test']}")

    elif choice == "3":
        # Her ikisi de
        print("\n🎯 Her iki dataset de oluşturuluyor...")
        results = dm.create_both_datasets(
            use_augmentation=True,
            yamnet_augmentation_factor=2,
            features_augmentation_factor=3,
        )

        print("\n✅ Dataset oluşturma tamamlandı!")

        # Detaylı bilgi
        if results.get("yamnet"):
            config_yamnet = dm.get_training_config()
            print(f"\n📊 YAMNet Training Config:")
            print(f"  Train samples: {config_yamnet['split_info']['train_samples']}")
            print(f"  Val samples: {config_yamnet['split_info']['val_samples']}")
            print(f"  Test samples: {config_yamnet['split_info']['test_samples']}")

    else:
        print("❌ Geçersiz seçim!")


if __name__ == "__main__":
    main()
