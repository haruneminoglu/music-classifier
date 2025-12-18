# src/audio_processing/classification_processor.py

import librosa
import numpy as np
import soundfile as sf
from scipy import signal
from typing import Dict, Any, Tuple, Optional
import os
import warnings


class ClassificationProcessor:
    """
    Random Forest tabanlı enstrüman sınıflandırması için ses işleme sınıfı
    Traditional audio features (MFCC, spektral özellikler) için optimize edilmiş
    """

    def __init__(self, sample_rate: int = 22050, duration_limit: int = 120):
        """
        ClassificationProcessor sınıfı başlatıcısı

        Args:
            sample_rate (int): Hedef örnekleme frekansı (Hz) - 22050 Hz önerilir
            duration_limit (int): Maksimum ses süresi (saniye)
        """
        self.sample_rate = sample_rate
        self.duration_limit = duration_limit
        self.supported_formats = {
            ".wav",
            ".mp3",
            ".flac",
            ".m4a",
            ".aac",
            ".aiff",
            ".au",
        }

        print(f"🎵 ClassificationProcessor başlatıldı")
        print(f"  Sample Rate: {self.sample_rate} Hz")
        print(f"  Duration Limit: {self.duration_limit} saniye")

    def load_audio(
        self, file_path: str, mono: bool = True
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Ses dosyasını yükler ve temel bilgileri döndürür

        Args:
            file_path (str): Ses dosyasının yolu
            mono (bool): Mono kanala çevirilip çevrilmeyeceği

        Returns:
            Tuple[np.ndarray, Dict]: (ses_verisi, metadata)

        Raises:
            FileNotFoundError: Dosya bulunamadığında
            ValueError: Desteklenmeyen format veya bozuk dosya
            RuntimeError: Yükleme hatası
        """
        # Dosya varlığı kontrolü
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Ses dosyası bulunamadı: {file_path}")

        # Format kontrolü
        file_ext = os.path.splitext(file_path)[1].lower()
        if file_ext not in self.supported_formats:
            raise ValueError(
                f"Desteklenmeyen format: {file_ext}. Desteklenen: {self.supported_formats}"
            )

        try:
            # Önce orijinal dosya bilgilerini al
            try:
                info = sf.info(file_path)
                original_sr = info.samplerate
                original_duration = info.duration
            except:
                original_sr = None
                original_duration = None

            # Ses dosyasını yükle ve resample yap
            audio_data, loaded_sr = librosa.load(
                file_path,
                sr=self.sample_rate,  # Hedef sample rate
                mono=mono,
                duration=self.duration_limit,
            )

            # Metadata bilgileri
            metadata = {
                "original_sample_rate": original_sr if original_sr else "unknown",
                "target_sample_rate": self.sample_rate,
                "sample_rate": self.sample_rate,
                "original_duration": (
                    original_duration if original_duration else "unknown"
                ),
                "duration": len(audio_data) / self.sample_rate,
                "samples": len(audio_data),
                "file_path": file_path,
                "file_name": os.path.basename(file_path),
                "file_size": os.path.getsize(file_path),
                "format": file_ext,
                "is_mono": mono,
            }

            return audio_data, metadata

        except Exception as e:
            raise RuntimeError(
                f"Ses dosyası yüklenirken hata ({os.path.basename(file_path)}): {str(e)}"
            )

    def get_audio_info(self, file_path: str) -> Dict[str, Any]:
        """
        Dosyayı yüklemeden temel bilgileri alır

        Args:
            file_path (str): Ses dosyası yolu

        Returns:
            Dict: Dosya bilgileri
        """
        try:
            info = sf.info(file_path)

            return {
                "duration": info.duration,
                "sample_rate": info.samplerate,
                "channels": info.channels,
                "frames": info.frames,
                "format": info.format,
                "subtype": info.subtype,
                "file_size": os.path.getsize(file_path),
                "needs_resampling": info.samplerate != self.sample_rate,
            }
        except Exception as e:
            return {"error": str(e)}

    def normalize_audio(
        self, audio_data: np.ndarray, method: str = "peak"
    ) -> np.ndarray:
        """
        Ses verisini normalize eder

        Args:
            audio_data: Ses verisi
            method: 'peak' (maksimum değere göre) veya 'rms' (RMS'e göre)

        Returns:
            np.ndarray: Normalize edilmiş ses verisi
        """
        if method == "peak":
            # Peak normalization
            max_val = np.max(np.abs(audio_data))
            if max_val > 0:
                return audio_data / max_val
            return audio_data

        elif method == "rms":
            # RMS normalization
            rms = np.sqrt(np.mean(audio_data**2))
            if rms > 0:
                # Target RMS değeri
                target_rms = 0.1
                return audio_data * (target_rms / rms)
            return audio_data

        else:
            return audio_data

    def noise_reduction(
        self, audio_data: np.ndarray, cutoff_freq: float = 80.0
    ) -> np.ndarray:
        """
        Basit gürültü azaltma (high-pass filter)

        Args:
            audio_data: Ses verisi
            cutoff_freq: Kesim frekansı (Hz)

        Returns:
            np.ndarray: Filtrelenmiş ses verisi
        """
        # Belirtilen frekans altındaki frekansları filtrele
        sos = signal.butter(
            5, cutoff_freq, btype="high", fs=self.sample_rate, output="sos"
        )
        return signal.sosfilt(sos, audio_data)

    def trim_silence(
        self, audio_data: np.ndarray, top_db: int = 20
    ) -> np.ndarray:
        """
        Başlangıç ve sondaki sessizlikleri kırpar

        Args:
            audio_data: Ses verisi
            top_db: Sessizlik eşiği (dB)

        Returns:
            np.ndarray: Kırpılmış ses verisi
        """
        trimmed, _ = librosa.effects.trim(audio_data, top_db=top_db)
        return trimmed

    def apply_window(
        self, audio_data: np.ndarray, window_type: str = "hann"
    ) -> np.ndarray:
        """
        Ses verisine pencere fonksiyonu uygular

        Args:
            audio_data: Ses verisi
            window_type: 'hann', 'hamming', veya 'blackman'

        Returns:
            np.ndarray: Pencere uygulanmış ses verisi
        """
        if window_type == "hann":
            window = np.hanning(len(audio_data))
        elif window_type == "hamming":
            window = np.hamming(len(audio_data))
        elif window_type == "blackman":
            window = np.blackman(len(audio_data))
        else:
            return audio_data

        return audio_data * window

    def compute_stft(
        self, audio_data: np.ndarray, hop_length: int = None, n_fft: int = 2048
    ) -> np.ndarray:
        """
        Short-Time Fourier Transform hesaplar

        Args:
            audio_data: Ses verisi
            hop_length: Hop uzunluğu (None ise n_fft // 4)
            n_fft: FFT pencere boyutu

        Returns:
            np.ndarray: STFT matris
        """
        if hop_length is None:
            hop_length = n_fft // 4

        return librosa.stft(audio_data, n_fft=n_fft, hop_length=hop_length)

    def compute_spectrogram(
        self, audio_data: np.ndarray, **kwargs
    ) -> np.ndarray:
        """
        Spektrogram hesaplar

        Args:
            audio_data: Ses verisi
            **kwargs: STFT parametreleri

        Returns:
            np.ndarray: Spektrogram
        """
        stft = self.compute_stft(audio_data, **kwargs)
        return np.abs(stft)

    def split_audio_segments(
        self, audio_data: np.ndarray, segment_length: float, overlap: float = 0.0
    ) -> list:
        """
        Ses verisini belirli uzunluktaki segmentlere böler

        Args:
            audio_data: Ses verisi
            segment_length: Segment uzunluğu (saniye)
            overlap: Segment overlap (saniye)

        Returns:
            list: Ses segmentleri listesi
        """
        segment_samples = int(segment_length * self.sample_rate)
        overlap_samples = int(overlap * self.sample_rate)
        step_samples = segment_samples - overlap_samples

        segments = []

        for i in range(0, len(audio_data) - segment_samples + 1, step_samples):
            segment = audio_data[i : i + segment_samples]
            if len(segment) >= segment_samples:
                segments.append(segment)

        # Son segment'i de ekle (eğer yeterince uzunsa)
        if len(audio_data) > segment_samples:
            last_segment = audio_data[-segment_samples:]
            if len(last_segment) >= segment_samples and last_segment.tolist() not in [
                s.tolist() for s in segments
            ]:
                segments.append(last_segment)

        return segments

    def validate_audio_data(self, audio_data: np.ndarray) -> bool:
        """
        Ses verisinin geçerliliğini kontrol eder

        Args:
            audio_data: Ses verisi

        Returns:
            bool: Geçerli ise True

        Raises:
            ValueError: Geçersiz veri durumunda
        """
        if len(audio_data) == 0:
            raise ValueError("Boş ses verisi")

        if np.all(audio_data == 0):
            warnings.warn("Ses verisi tamamen sessiz")

        if np.any(np.isnan(audio_data)) or np.any(np.isinf(audio_data)):
            raise ValueError("Ses verisinde geçersiz değerler (NaN veya Inf)")

        return True

    def preprocess(
        self,
        audio_data: np.ndarray,
        normalize_method: str = "rms",
        apply_noise_reduction: bool = True,
        apply_trim: bool = True,
        trim_db: int = 20,
    ) -> np.ndarray:
        """
        Random Forest sınıflandırması için ses verisini ön işleme tabi tutar

        Args:
            audio_data: Ham ses verisi
            normalize_method: 'peak' veya 'rms' (önerilen: 'rms')
            apply_noise_reduction: Gürültü azaltma uygulansın mı
            apply_trim: Sessizlik kırpma uygulansın mı
            trim_db: Sessizlik kırpma eşiği (dB)

        Returns:
            np.ndarray: İşlenmiş ses verisi
        """
        # Boş veri kontrolü
        self.validate_audio_data(audio_data)

        # Normalizasyon
        audio_data = self.normalize_audio(audio_data, method=normalize_method)

        # Gürültü azaltma (isteğe bağlı)
        if apply_noise_reduction:
            audio_data = self.noise_reduction(audio_data)

        # Sessizlik kaldırma (isteğe bağlı)
        if apply_trim:
            audio_data = self.trim_silence(audio_data, top_db=trim_db)

        # Final normalization
        audio_data = self.normalize_audio(audio_data, method="peak")

        # Clip to [-1, 1]
        audio_data = np.clip(audio_data, -1.0, 1.0)

        return audio_data


# Geriye uyumluluk için alias
AudioProcessor = ClassificationProcessor


def test_classification_processor():
    """ClassificationProcessor sınıfını test eder"""
    print("=" * 60)
    print("ClassificationProcessor Test (Random Forest)")
    print("=" * 60)

    # Test 1: Processor oluşturma
    print("\n📊 Test 1: Processor Oluşturma")
    processor = ClassificationProcessor(sample_rate=22050)
    print(f"  Sample Rate: {processor.sample_rate} Hz")
    print(f"  Desteklenen formatlar: {len(processor.supported_formats)} format")

    # Test 2: Örnek ses verisi ile preprocessing
    print("\n📊 Test 2: Preprocessing Test")
    sample_audio = np.random.randn(22050 * 3)  # 3 saniye 22050 Hz

    print("  Ham ses:")
    print(f"    Shape: {sample_audio.shape}")
    print(f"    Range: [{sample_audio.min():.3f}, {sample_audio.max():.3f}]")

    processed = processor.preprocess(
        sample_audio,
        normalize_method="rms",
        apply_noise_reduction=True,
        apply_trim=True
    )
    print("  İşlenmiş ses:")
    print(f"    Shape: {processed.shape}")
    print(f"    Range: [{processed.min():.3f}, {processed.max():.3f}]")

    # Test 3: Segment bölme
    print("\n📊 Test 3: Segment Bölme")
    segments = processor.split_audio_segments(
        sample_audio, segment_length=1.0, overlap=0.5
    )
    print(f"  {len(segments)} segment oluşturuldu")
    if segments:
        print(f"  Her segment: {len(segments[0])} sample")

    # Test 4: STFT ve Spektrogram
    print("\n📊 Test 4: STFT ve Spektrogram")
    stft = processor.compute_stft(sample_audio)
    spectrogram = processor.compute_spectrogram(sample_audio)
    print(f"  STFT shape: {stft.shape}")
    print(f"  Spektrogram shape: {spectrogram.shape}")

    # Test 5: Normalizasyon methodları
    print("\n📊 Test 5: Normalizasyon Karşılaştırma")
    peak_norm = processor.normalize_audio(sample_audio, method="peak")
    rms_norm = processor.normalize_audio(sample_audio, method="rms")
    print(f"  Peak norm range: [{peak_norm.min():.3f}, {peak_norm.max():.3f}]")
    print(f"  RMS norm range: [{rms_norm.min():.3f}, {rms_norm.max():.3f}]")

    print("\n✅ Tüm testler başarılı!")
    print("=" * 60)


def demo_feature_extraction_pipeline():
    """Random Forest için tipik bir feature extraction pipeline'ı gösterir"""
    print("\n" + "=" * 60)
    print("Random Forest Pipeline Demo")
    print("=" * 60)

    processor = ClassificationProcessor(sample_rate=22050)
    
    # Örnek ses verisi (3 saniye)
    sample_audio = np.random.randn(22050 * 3)
    
    print("\n1️⃣ Ses Yükleme ve Preprocessing")
    processed_audio = processor.preprocess(
        sample_audio,
        normalize_method="rms",
        apply_noise_reduction=True,
        apply_trim=True
    )
    print(f"  Processed audio: {len(processed_audio)} samples")
    
    print("\n2️⃣ Spektrogram Hesaplama")
    spectrogram = processor.compute_spectrogram(processed_audio, n_fft=2048)
    print(f"  Spectrogram shape: {spectrogram.shape}")
    
    print("\n3️⃣ Segment Bölme (eğer gerekirse)")
    segments = processor.split_audio_segments(
        processed_audio, 
        segment_length=2.0, 
        overlap=0.5
    )
    print(f"  {len(segments)} segment oluşturuldu")
    
    print("\n4️⃣ Feature Extraction için hazır!")
    print("  Bu noktada MFCC, spektral özellikler vb. çıkarılabilir")
    print("  Daha sonra Random Forest modeline verilebilir")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    test_classification_processor()
    demo_feature_extraction_pipeline()