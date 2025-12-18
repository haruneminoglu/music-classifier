"""
Musical Instrument Classifier Desktop App
Basit masaüstü uygulaması

Kullanım:
    python instrument_classifier_app.py

Gereksinimler:
    pip install tkinter pillow
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
import sys
import json
import pickle
import numpy as np
from typing import Dict, Optional, List
import traceback

# Proje modüllerini import et
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    import tensorflow as tf
    import tensorflow_hub as hub
    import librosa
    import joblib
    from autogluon.tabular import TabularPredictor
    import pandas as pd

    # Proje modülleri
    from src.audio_processing.classification_processor import ClassificationProcessor
    from src.feature_extraction.classification_extractor import ClassificationExtractor

    DEPENDENCIES_OK = True
except ImportError as e:
    DEPENDENCIES_OK = False
    IMPORT_ERROR = str(e)


class InstrumentClassifierApp:
    """Masaüstü uygulama ana sınıfı"""

    def __init__(self, root):
        self.root = root
        self.root.title("🎵 Musical Instrument Classifier")
        self.root.geometry("900x700")
        self.root.resizable(True, True)

        # Değişkenler
        self.audio_file = None
        self.selected_model_type = tk.StringVar(value="1")
        self.instruments = ["cello", "clarinet", "flute", "trumpet", "violin"]

        # Model cache
        self.yamnet_model = None
        self.loaded_models = {}

        # Proje dizinleri
        self.project_root = project_root
        self.models_dir = self.project_root / "models"

        # UI oluştur
        if DEPENDENCIES_OK:
            self.create_ui()
        else:
            self.show_dependency_error()

    def show_dependency_error(self):
        """Bağımlılık hatası göster"""
        frame = ttk.Frame(self.root, padding=20)
        frame.pack(fill="both", expand=True)

        ttk.Label(
            frame,
            text="❌ Gerekli kütüphaneler eksik!",
            font=("Arial", 16, "bold"),
            foreground="red",
        ).pack(pady=20)

        error_text = tk.Text(frame, height=15, width=80, wrap="word")
        error_text.pack(pady=10)
        error_text.insert("1.0", f"Import Hatası:\n{IMPORT_ERROR}\n\n")
        error_text.insert("end", "Gerekli kütüphaneleri yükleyin:\n\n")
        error_text.insert("end", "pip install tensorflow tensorflow-hub librosa\n")
        error_text.insert("end", "pip install scikit-learn joblib\n")
        error_text.insert("end", "pip install autogluon\n")
        error_text.config(state="disabled")

    def create_ui(self):
        """Ana UI'ı oluştur"""
        # Ana konteyner
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill="both", expand=True)

        # Başlık
        title_frame = ttk.Frame(main_frame)
        title_frame.pack(fill="x", pady=(0, 15))

        ttk.Label(
            title_frame,
            text="🎵 Musical Instrument Classifier",
            font=("Arial", 18, "bold"),
        ).pack()

        ttk.Label(
            title_frame,
            text="Ses dosyalarını analiz ederek enstrüman tahminleri yapın",
            font=("Arial", 10),
        ).pack()

        # Sol panel - Dosya seçimi ve model seçimi
        left_frame = ttk.LabelFrame(main_frame, text="⚙️ Ayarlar", padding=15)
        left_frame.pack(side="left", fill="both", expand=False, padx=(0, 10))

        self.create_file_selection(left_frame)
        self.create_model_selection(left_frame)
        self.create_action_buttons(left_frame)

        # Sağ panel - Sonuçlar
        right_frame = ttk.LabelFrame(main_frame, text="📊 Sonuçlar", padding=15)
        right_frame.pack(side="right", fill="both", expand=True)

        self.create_results_panel(right_frame)

    def create_file_selection(self, parent):
        """Dosya seçim bölümü"""
        file_frame = ttk.LabelFrame(parent, text="📁 Ses Dosyası", padding=10)
        file_frame.pack(fill="x", pady=(0, 15))

        self.file_label = ttk.Label(
            file_frame, text="Dosya seçilmedi", foreground="gray", wraplength=300
        )
        self.file_label.pack(pady=5)

        ttk.Button(
            file_frame, text="🔍 Dosya Seç", command=self.select_audio_file
        ).pack(pady=5)

    def create_model_selection(self, parent):
        """Model seçim bölümü"""
        model_frame = ttk.LabelFrame(parent, text="🤖 Model Seçimi", padding=10)
        model_frame.pack(fill="x", pady=(0, 15))

        models = [
            ("1", "Random Forest (Traditional)"),
            ("2", "YAMNet Transfer Learning"),
            ("3", "AutoGluon (Traditional)"),
            ("4", "AutoGluon (YAMNet)"),
            ("5", "🔥 TÜM MODELLER (Karşılaştırma)"),
        ]

        for value, text in models:
            ttk.Radiobutton(
                model_frame, text=text, variable=self.selected_model_type, value=value
            ).pack(anchor="w", pady=3)

    def create_action_buttons(self, parent):
        """İşlem butonları"""
        button_frame = ttk.Frame(parent)
        button_frame.pack(fill="x", pady=(10, 0))

        self.predict_button = ttk.Button(
            button_frame,
            text="🎯 Tahmin Yap",
            command=self.run_prediction,
            state="disabled",
        )
        self.predict_button.pack(fill="x", pady=5)

        ttk.Button(button_frame, text="🔄 Temizle", command=self.clear_results).pack(
            fill="x", pady=5
        )

    def create_results_panel(self, parent):
        """Sonuçlar paneli"""
        # Scrollable text widget
        text_frame = ttk.Frame(parent)
        text_frame.pack(fill="both", expand=True)

        scrollbar = ttk.Scrollbar(text_frame)
        scrollbar.pack(side="right", fill="y")

        self.results_text = tk.Text(
            text_frame,
            wrap="word",
            yscrollcommand=scrollbar.set,
            font=("Consolas", 10),
            bg="#f5f5f5",
        )
        self.results_text.pack(side="left", fill="both", expand=True)
        scrollbar.config(command=self.results_text.yview)

        # Text tags for formatting
        self.results_text.tag_config(
            "header", font=("Arial", 12, "bold"), foreground="blue"
        )
        self.results_text.tag_config(
            "success", foreground="green", font=("Arial", 10, "bold")
        )
        self.results_text.tag_config(
            "error", foreground="red", font=("Arial", 10, "bold")
        )
        self.results_text.tag_config(
            "instrument", font=("Arial", 14, "bold"), foreground="darkblue"
        )

    def select_audio_file(self):
        """Ses dosyası seç"""
        filetypes = [
            ("Ses Dosyaları", "*.wav *.mp3 *.flac *.m4a *.aiff"),
            ("Tüm Dosyalar", "*.*"),
        ]

        filename = filedialog.askopenfilename(
            title="Ses Dosyası Seçin", filetypes=filetypes
        )

        if filename:
            self.audio_file = Path(filename)
            self.file_label.config(text=self.audio_file.name, foreground="black")
            self.predict_button.config(state="normal")
            self.log_result(f"✅ Dosya seçildi: {self.audio_file.name}\n", "success")

    def run_prediction(self):
        """Tahmin işlemini başlat"""
        if not self.audio_file:
            messagebox.showwarning("Uyarı", "Lütfen önce bir ses dosyası seçin!")
            return

        self.clear_results()
        self.log_result("🚀 Tahmin başlatılıyor...\n\n", "header")
        self.predict_button.config(state="disabled")
        self.root.update()

        try:
            model_type = self.selected_model_type.get()

            if model_type == "5":
                # Tüm modelleri karşılaştır
                self.compare_all_models()
            else:
                # Tek model tahmini
                self.predict_single_model(model_type)

        except Exception as e:
            self.log_result(f"❌ HATA: {str(e)}\n\n", "error")
            self.log_result(traceback.format_exc())

        finally:
            self.predict_button.config(state="normal")

    def predict_single_model(self, model_type: str):
        """Tek model ile tahmin"""
        model_info = {
            "1": {
                "name": "Random Forest (Traditional)",
                "method": self.predict_random_forest,
            },
            "2": {"name": "YAMNet Transfer Learning", "method": self.predict_yamnet},
            "3": {
                "name": "AutoGluon (Traditional)",
                "method": self.predict_autogluon_traditional,
            },
            "4": {
                "name": "AutoGluon (YAMNet)",
                "method": self.predict_autogluon_yamnet,
            },
        }

        info = model_info[model_type]
        self.log_result(f"🤖 Model: {info['name']}\n", "header")
        self.log_result(f"📁 Dosya: {self.audio_file.name}\n\n")

        result = info["method"]()

        if result and "error" not in result:
            self.display_prediction(result)
        elif result:
            self.log_result(f"❌ {result['error']}\n", "error")

    def compare_all_models(self):
        """Tüm modelleri karşılaştır"""
        self.log_result("🔥 TÜM MODELLER KARŞILAŞTIRMASI\n", "header")
        self.log_result(f"📁 Dosya: {self.audio_file.name}\n")
        self.log_result("=" * 60 + "\n\n")

        models = [
            ("Random Forest", self.predict_random_forest),
            ("YAMNet Transfer", self.predict_yamnet),
            ("AutoGluon Traditional", self.predict_autogluon_traditional),
            ("AutoGluon YAMNet", self.predict_autogluon_yamnet),
        ]

        results = []

        for name, method in models:
            self.log_result(f"📊 {name}...\n")
            self.root.update()

            try:
                result = method()
                if result and "error" not in result:
                    results.append((name, result))
                    pred = result["predicted_instrument"]
                    conf = result["confidence"]
                    self.log_result(f"   ✅ Tahmin: {pred} ({conf:.1%})\n\n")
                else:
                    self.log_result(
                        f"   ❌ Hata: {result.get('error', 'Unknown')}\n\n", "error"
                    )
            except Exception as e:
                self.log_result(f"   ❌ Hata: {str(e)}\n\n", "error")

        # Karşılaştırma tablosu
        if results:
            self.log_result("\n" + "=" * 60 + "\n", "header")
            self.log_result("📊 SONUÇ TABLOSU\n", "header")
            self.log_result("=" * 60 + "\n\n")

            for name, result in results:
                pred = result["predicted_instrument"]
                conf = result["confidence"]
                self.log_result(f"{name:25} → {pred:12} ({conf:.1%})\n")

            # Konsensüs
            predictions = [r[1]["predicted_instrument"] for r in results]
            from collections import Counter

            consensus = Counter(predictions).most_common(1)[0]

            self.log_result(
                f"\n🎯 En Çok Tahmin: {consensus[0]} ({consensus[1]}/{len(results)} model)\n",
                "success",
            )

    def predict_random_forest(self) -> Dict:
        """Random Forest modeli ile tahmin"""
        try:
            model_dir = self.models_dir / "random_forest_good_sounds"

            if not model_dir.exists():
                return {"error": f"Model dizini bulunamadı: {model_dir}"}

            # Model dosyalarını bul
            model_file = list(model_dir.glob("*_model.pkl"))
            scaler_file = list(model_dir.glob("*_scaler.pkl"))
            encoder_file = list(model_dir.glob("*_encoder.pkl"))

            if not (model_file and scaler_file and encoder_file):
                return {"error": "Model dosyaları eksik!"}

            # Model yükle
            model = joblib.load(model_file[0])
            scaler = joblib.load(scaler_file[0])
            encoder = joblib.load(encoder_file[0])

            # Feature extraction (22kHz)
            processor = ClassificationProcessor(sample_rate=22050)
            extractor = ClassificationExtractor(sample_rate=22050)

            audio_data, _ = processor.load_audio(str(self.audio_file))
            processed = processor.preprocess(audio_data)
            features = extractor.extract_features(processed)

            # Reshape ve scale
            features_scaled = scaler.transform(features.reshape(1, -1))

            # Tahmin
            pred_idx = model.predict(features_scaled)[0]
            predicted = encoder.inverse_transform([pred_idx])[0]

            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(features_scaled)[0]
                confidence = proba[pred_idx]
                probabilities = {
                    instr: float(prob) for instr, prob in zip(encoder.classes_, proba)
                }
            else:
                confidence = 1.0
                probabilities = {predicted: 1.0}

            return {
                "predicted_instrument": predicted,
                "confidence": float(confidence),
                "probabilities": probabilities,
            }

        except Exception as e:
            return {"error": str(e)}

    def predict_yamnet(self) -> Dict:
        """YAMNet Transfer Learning ile tahmin"""
        try:
            model_dir = self.models_dir / "yamnet_good_sounds"

            if not model_dir.exists():
                return {"error": f"Model dizini bulunamadı: {model_dir}"}

            # YAMNet yükle
            if self.yamnet_model is None:
                self.yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")

            # Classification head yükle
            head_file = list(model_dir.glob("*_head.keras"))
            encoder_file = list(model_dir.glob("*_label_encoder.pkl"))

            if not (head_file and encoder_file):
                return {"error": "Model dosyaları eksik!"}

            classification_head = tf.keras.models.load_model(head_file[0])

            with open(encoder_file[0], "rb") as f:
                encoder = pickle.load(f)

            # Ses yükle (16kHz)
            waveform, _ = librosa.load(str(self.audio_file), sr=16000, mono=True)
            waveform = waveform / (np.max(np.abs(waveform)) + 1e-8)

            # YAMNet embedding
            _, embeddings, _ = self.yamnet_model(waveform)
            avg_embedding = tf.reduce_mean(embeddings, axis=0, keepdims=True)

            # Tahmin
            proba = classification_head.predict(avg_embedding, verbose=0)[0]
            pred_idx = np.argmax(proba)
            predicted = encoder.inverse_transform([pred_idx])[0]

            return {
                "predicted_instrument": predicted,
                "confidence": float(proba[pred_idx]),
                "probabilities": {
                    instr: float(prob) for instr, prob in zip(encoder.classes_, proba)
                },
            }

        except Exception as e:
            return {"error": str(e)}

    def predict_autogluon_traditional(self) -> Dict:
        """AutoGluon Traditional Features ile tahmin"""
        try:
            model_dir = self.models_dir / "autogluon_traditional"

            if not model_dir.exists():
                return {"error": f"Model dizini bulunamadı: {model_dir}"}

            # Model yükle
            predictor = TabularPredictor.load(str(model_dir))

            # Feature extraction (22kHz)
            processor = ClassificationProcessor(sample_rate=22050)
            extractor = ClassificationExtractor(sample_rate=22050)

            audio_data, _ = processor.load_audio(str(self.audio_file))
            processed = processor.preprocess(audio_data)
            features = extractor.extract_features(processed)

            # DataFrame oluştur
            feature_names = [f"feature_{i}" for i in range(len(features))]
            df = pd.DataFrame([features], columns=feature_names)

            # Tahmin
            prediction = predictor.predict(df)[0]
            probabilities = predictor.predict_proba(df).iloc[0].to_dict()

            return {
                "predicted_instrument": prediction,
                "confidence": float(probabilities[prediction]),
                "probabilities": probabilities,
            }

        except Exception as e:
            return {"error": str(e)}

    def predict_autogluon_yamnet(self) -> Dict:
        """AutoGluon YAMNet Embeddings ile tahmin"""
        try:
            model_dir = self.models_dir / "autogluon_yamnet"

            if not model_dir.exists():
                return {"error": f"Model dizini bulunamadı: {model_dir}"}

            # YAMNet yükle
            if self.yamnet_model is None:
                self.yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")

            # Model yükle
            predictor = TabularPredictor.load(str(model_dir))

            # Ses yükle (16kHz)
            waveform, _ = librosa.load(str(self.audio_file), sr=16000, mono=True)
            waveform = waveform / (np.max(np.abs(waveform)) + 1e-8)

            # YAMNet embedding
            _, embeddings, _ = self.yamnet_model(waveform)
            avg_embedding = tf.reduce_mean(embeddings, axis=0).numpy()

            # DataFrame oluştur
            feature_names = [f"embedding_{i}" for i in range(len(avg_embedding))]
            df = pd.DataFrame([avg_embedding], columns=feature_names)

            # Tahmin
            prediction = predictor.predict(df)[0]
            probabilities = predictor.predict_proba(df).iloc[0].to_dict()

            return {
                "predicted_instrument": prediction,
                "confidence": float(probabilities[prediction]),
                "probabilities": probabilities,
            }

        except Exception as e:
            return {"error": str(e)}

    def display_prediction(self, result: Dict):
        """Tahmin sonucunu göster"""
        pred = result["predicted_instrument"]
        conf = result["confidence"]
        probs = result["probabilities"]

        self.log_result("=" * 60 + "\n", "header")
        self.log_result("🎯 TAHMİN SONUCU\n", "header")
        self.log_result("=" * 60 + "\n\n")

        self.log_result(f"🎺 Enstrüman: ", "success")
        self.log_result(f"{pred.upper()}\n", "instrument")
        self.log_result(f"💯 Güven: {conf:.1%}\n\n", "success")

        self.log_result("📊 Tüm Olasılıklar:\n", "header")
        self.log_result("-" * 40 + "\n")

        sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        for instrument, prob in sorted_probs:
            bar_length = int(prob * 30)
            bar = "█" * bar_length
            self.log_result(f"{instrument:12} {bar:30} {prob:.1%}\n")

    def log_result(self, text: str, tag: str = None):
        """Sonuç paneline log ekle"""
        if tag:
            self.results_text.insert("end", text, tag)
        else:
            self.results_text.insert("end", text)

        self.results_text.see("end")
        self.root.update()

    def clear_results(self):
        """Sonuçları temizle"""
        self.results_text.delete("1.0", "end")


def main():
    """Ana program"""
    root = tk.Tk()
    app = InstrumentClassifierApp(root)

    # Merkeze al
    root.update_idletasks()
    width = root.winfo_width()
    height = root.winfo_height()
    x = (root.winfo_screenwidth() // 2) - (width // 2)
    y = (root.winfo_screenheight() // 2) - (height // 2)
    root.geometry(f"{width}x{height}+{x}+{y}")

    root.mainloop()


if __name__ == "__main__":
    main()
