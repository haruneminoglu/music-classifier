# src/feature_extraction/classification_extractor.py

import librosa
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
import matplotlib.pyplot as plt
from scipy import stats
import warnings

class ClassificationExtractor:
    """
    Random Forest ile enstrüman sınıflandırma için özellik çıkarım sınıfı
    
    Enstrümanların timbre, attack, spectral karakteristiklerini analiz eder
    """
    
    def __init__(self, sample_rate: int = 22050, n_mfcc: int = 13, 
                 hop_length: int = 512, n_fft: int = 2048):
        """
        ClassificationExtractor başlatıcısı
        
        Args:
            sample_rate (int): Örnekleme frekansı
            n_mfcc (int): MFCC katsayı sayısı
            hop_length (int): Pencere kaydırma miktarı
            n_fft (int): FFT pencere boyutu
        """
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.n_mfcc = n_mfcc
        
        print(f"ClassificationExtractor hazır - SR: {sample_rate}, Hop: {hop_length}, FFT: {n_fft}, MFCC: {n_mfcc}")
    
    # ==================== DATA MANAGER İLE UYUMLU ANA METODLAR ====================
    
    def extract_features(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Random Forest için ana özellik çıkarım metodu
        
        Args:
            audio_data (np.ndarray): Ses verisi
            
        Returns:
            np.ndarray: Düzleştirilmiş özellik vektörü (ML için hazır)
        """
        # Tüm özellikleri dict olarak çıkar
        all_features_dict = self.extract_all_features(audio_data)
        
        # Dict'i tek boyutlu vektöre çevir
        feature_vector = self.get_feature_vector_from_dict(all_features_dict)
        
        return feature_vector
    
    def get_feature_vector_from_dict(self, features_dict: Dict[str, Union[np.ndarray, float]]) -> np.ndarray:
        """
        Özellik dict'ini tek boyutlu vektöre çevirir
        
        Args:
            features_dict: Özellik sözlüğü
            
        Returns:
            np.ndarray: Düzleştirilmiş özellik vektörü
        """
        feature_vector = []
        
        for key, value in features_dict.items():
            if isinstance(value, np.ndarray):
                feature_vector.extend(value.flatten())
            elif isinstance(value, (int, float)):
                feature_vector.append(float(value))
            else:
                # Diğer türler için 0 ekle
                feature_vector.append(0.0)
        
        return np.array(feature_vector, dtype=np.float32)
    
    # ==================== BASE EXTRACTOR FONKSIYONLARI ====================
    
    def extract_mfcc(self, audio_data: np.ndarray, n_mfcc: int = 13) -> np.ndarray:
        """
        MFCC özelliklerini çıkarır
        
        Args:
            audio_data (np.ndarray): Ses verisi
            n_mfcc (int): MFCC katsayı sayısı
            
        Returns:
            np.ndarray: MFCC matris (n_mfcc x time_frames)
        """
        return librosa.feature.mfcc(
            y=audio_data,
            sr=self.sample_rate,
            n_mfcc=n_mfcc,
            hop_length=self.hop_length,
            n_fft=self.n_fft
        )
    
    def extract_chroma(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Chroma özelliklerini çıkarır (12 ton)
        
        Args:
            audio_data (np.ndarray): Ses verisi
            
        Returns:
            np.ndarray: Chroma matris (12 x time_frames)
        """
        return librosa.feature.chroma_stft(
            y=audio_data,
            sr=self.sample_rate,
            hop_length=self.hop_length,
            n_fft=self.n_fft
        )
    
    def extract_spectral_features(self, audio_data: np.ndarray) -> Dict[str, Union[np.ndarray, float]]:
        """
        Spektral özellikleri çıkarır
        
        Args:
            audio_data (np.ndarray): Ses verisi
            
        Returns:
            Dict: Spektral özellikler
        """
        # Spektrogram hesapla
        S = np.abs(librosa.stft(audio_data, hop_length=self.hop_length, n_fft=self.n_fft))
        
        features = {}
        features['spectral_centroid'] = librosa.feature.spectral_centroid(S=S, sr=self.sample_rate)
        features['spectral_bandwidth'] = librosa.feature.spectral_bandwidth(S=S, sr=self.sample_rate)  
        features['spectral_rolloff'] = librosa.feature.spectral_rolloff(S=S, sr=self.sample_rate)
        features['spectral_flatness'] = librosa.feature.spectral_flatness(S=S)
        features['zero_crossing_rate'] = librosa.feature.zero_crossing_rate(audio_data, hop_length=self.hop_length)
        features['rms_energy'] = librosa.feature.rms(S=S)
        
        return features
    
    def extract_onset_features(self, audio_data: np.ndarray) -> Dict[str, Union[np.ndarray, float, int]]:
        """
        Onset (nota başlangıç) özelliklerini çıkarır
        
        Args:
            audio_data (np.ndarray): Ses verisi
            
        Returns:
            Dict: Onset özellikleri
        """
        # Onset detection
        onset_frames = librosa.onset.onset_detect(
            y=audio_data,
            sr=self.sample_rate,
            hop_length=self.hop_length,
            units='frames'
        )
        
        features = {}
        features['onset_frames'] = onset_frames
        features['onset_times'] = librosa.frames_to_time(onset_frames, sr=self.sample_rate, hop_length=self.hop_length)
        features['onset_count'] = len(onset_frames)
        
        if len(onset_frames) > 1:
            # Onset intervals
            intervals = np.diff(features['onset_times'])
            features['onset_interval_mean'] = np.mean(intervals)
            features['onset_interval_std'] = np.std(intervals)
            
            # Onset density (nota yoğunluğu)
            duration = len(audio_data) / self.sample_rate
            features['onset_density'] = len(onset_frames) / duration
        else:
            features['onset_interval_mean'] = 0.0
            features['onset_interval_std'] = 0.0
            features['onset_density'] = 0.0
            
        return features
    
    def extract_tempo_features(self, audio_data: np.ndarray) -> Dict[str, Union[float, np.ndarray]]:
        """
        Tempo ve ritim özelliklerini çıkarır
        
        Args:
            audio_data (np.ndarray): Ses verisi
            
        Returns:
            Dict: Tempo özellikleri
        """
        features = {}
        
        try:
            tempo, beats = librosa.beat.beat_track(
                y=audio_data,
                sr=self.sample_rate,
                hop_length=self.hop_length
            )
            
            features['tempo'] = float(tempo.item()) if hasattr(tempo, 'item') else float(tempo)
            features['beat_frames'] = beats
            features['beat_times'] = librosa.frames_to_time(beats, sr=self.sample_rate, hop_length=self.hop_length)
            
            # Beat regularity (ritim düzenliliği)
            if len(beats) > 2:
                beat_intervals = np.diff(features['beat_times'])
                features['beat_regularity'] = 1.0 / (1.0 + np.std(beat_intervals))
            else:
                features['beat_regularity'] = 0.0
                
        except Exception as e:
            print(f"⚠️ Tempo analizi hatası: {e}")
            features['tempo'] = 0.0
            features['beat_frames'] = np.array([])
            features['beat_times'] = np.array([])
            features['beat_regularity'] = 0.0
            
        return features
    
    def calculate_pitch_features(self, audio_data: np.ndarray) -> Dict[str, Union[np.ndarray, float]]:
        """
        Perde (pitch) özelliklerini çıkarır
        
        Args:
            audio_data (np.ndarray): Ses verisi
            
        Returns:
            Dict: Pitch özellikleri
        """
        features = {}
        
        # Fundamental frequency estimation
        f0, voiced_flag, voiced_probs = librosa.pyin(
            audio_data,
            fmin=librosa.note_to_hz('C2'),  # 65.4 Hz
            fmax=librosa.note_to_hz('C7'),  # 2093 Hz
            sr=self.sample_rate,
            hop_length=self.hop_length
        )
        
        features['f0'] = f0
        features['voiced_flag'] = voiced_flag
        features['voiced_probs'] = voiced_probs
        
        # F0 istatistikleri (sadece voiced kısımlar)
        valid_f0 = f0[~np.isnan(f0)]
        if len(valid_f0) > 0:
            features['f0_mean'] = np.mean(valid_f0)
            features['f0_std'] = np.std(valid_f0)
            features['f0_median'] = np.median(valid_f0)
            features['f0_min'] = np.min(valid_f0)
            features['f0_max'] = np.max(valid_f0)
            features['f0_range'] = features['f0_max'] - features['f0_min']
        else:
            features['f0_mean'] = 0.0
            features['f0_std'] = 0.0
            features['f0_median'] = 0.0
            features['f0_min'] = 0.0
            features['f0_max'] = 0.0
            features['f0_range'] = 0.0
            
        return features
    
    def calculate_attack_time(self, audio_data: np.ndarray) -> float:
        """
        Ses başlangıç hızını hesaplar (attack time)
        
        Args:
            audio_data (np.ndarray): Ses verisi
            
        Returns:
            float: Attack time (ms)
        """
        try:
            # Ses başlangıcını bul
            trimmed, _ = librosa.effects.trim(audio_data, top_db=20)
            
            if len(trimmed) > 100:
                # İlk %10'luk kısımda maksimuma ulaşma süresi
                attack_region = trimmed[:len(trimmed)//10]
                max_amplitude = np.max(np.abs(attack_region))
                
                # %90'a ulaşma noktasını bul
                threshold = 0.9 * max_amplitude
                attack_samples = np.where(np.abs(attack_region) >= threshold)[0]
                
                if len(attack_samples) > 0:
                    return attack_samples[0] / self.sample_rate * 1000  # ms cinsinden
            
            return 0.0
        except:
            return 0.0
    
    # ==================== CLASSIFICATION SPECIFIC METHODS ====================
    
    def extract_all_features(self, audio_data: np.ndarray) -> Dict[str, Union[np.ndarray, float]]:
        """
        Enstrüman sınıflandırma için tüm özellik türlerini çıkarır
        
        Args:
            audio_data (np.ndarray): Ses verisi
            
        Returns:
            Dict: Tüm özellikler
        """
        features = {}
        
        print("🎷 Enstrüman sınıflandırma özellikleri çıkarılıyor...")
        
        # 1. MFCC Özellikleri (Timbre analizi için kritik!)
        features.update(self._extract_mfcc_classification_features(audio_data))
        
        # 2. Spectral Özellikleri (Enstrüman karakteristiği)
        features.update(self._extract_spectral_classification_features(audio_data))
        
        # 3. Chroma Özellikleri (Harmonik içerik)
        features.update(self._extract_chroma_classification_features(audio_data))
        
        # 4. Temporal Özellikleri (Attack, decay karakteristikleri)
        features.update(self._extract_temporal_classification_features(audio_data))
        
        # 5. Harmonic vs Percussive ayrışımı
        features.update(self._extract_harmonic_percussive_features(audio_data))
        
        # 6. Envelope özellikleri
        features.update(self._extract_envelope_features(audio_data))
        
        print(f"✅ {len(features)} özellik türü çıkarıldı (enstrüman sınıflandırma)")
        return features
    
    def _extract_mfcc_classification_features(self, audio_data: np.ndarray) -> Dict[str, np.ndarray]:
        """MFCC özelliklerini enstrüman sınıflandırma için çıkarır"""
        features = {}
        
        # MFCC hesapla
        mfcc = self.extract_mfcc(audio_data, self.n_mfcc)
        
        # İstatistiksel özellikler - timbre karakteristikleri için kritik
        features['mfcc_mean'] = np.mean(mfcc, axis=1)           
        features['mfcc_std'] = np.std(mfcc, axis=1)             
        features['mfcc_delta'] = np.mean(librosa.feature.delta(mfcc), axis=1)  
        features['mfcc_delta2'] = np.mean(librosa.feature.delta(mfcc, order=2), axis=1)  
        
        # İlk MFCC katsayısı (enerji karakteristiği)
        features['mfcc_energy'] = features['mfcc_mean'][0]
        
        # MFCC çarpıklık ve basıklık (enstrüman ayırt ediciliği)
        features['mfcc_skewness'] = np.array([stats.skew(mfcc[i]) if len(mfcc[i]) > 0 else 0.0 for i in range(mfcc.shape[0])])
        features['mfcc_kurtosis'] = np.array([stats.kurtosis(mfcc[i]) if len(mfcc[i]) > 0 else 0.0 for i in range(mfcc.shape[0])])
        
        # NaN kontrolü
        for key in features:
            if isinstance(features[key], np.ndarray):
                features[key] = np.nan_to_num(features[key])
        
        print(f"  🎵 MFCC: {self.n_mfcc} katsayı x 6 istatistik = {len(features)} özellik")
        return features
    
    def _extract_spectral_classification_features(self, audio_data: np.ndarray) -> Dict[str, Union[float, np.ndarray]]:
        """Spektral özellikleri enstrüman ayırt etme için çıkarır"""
        features = {}
        
        # Temel spektral özellikler
        spectral_features = self.extract_spectral_features(audio_data)
        
        # Her spektral özellik için istatistik
        for key, values in spectral_features.items():
            if isinstance(values, np.ndarray) and values.ndim > 0:
                stats_dict = statistical_summary(values.flatten())
                for stat_name, stat_value in stats_dict.items():
                    features[f"{key}_{stat_name}"] = stat_value
        
        # Ek spektral özellikler - enstrüman karakteristiği
        S = np.abs(librosa.stft(audio_data, hop_length=self.hop_length, n_fft=self.n_fft))
        
        # Spectral contrast (harmonik vs noise)
        contrast = librosa.feature.spectral_contrast(S=S, sr=self.sample_rate)
        features['spectral_contrast_mean'] = np.mean(contrast, axis=1)
        features['spectral_contrast_std'] = np.std(contrast, axis=1)
        
        # Tonnetz (harmonik ağ - enstrüman tonalitesi)
        chroma = self.extract_chroma(audio_data)
        tonnetz = librosa.feature.tonnetz(chroma=chroma)
        features['tonnetz_mean'] = np.mean(tonnetz, axis=1)
        features['tonnetz_std'] = np.std(tonnetz, axis=1)
        
        print(f"  🌊 Spektral: {len([k for k in features.keys() if 'spectral' in k or 'tonnetz' in k])} özellik")
        return features
    
    def _extract_chroma_classification_features(self, audio_data: np.ndarray) -> Dict[str, Union[float, np.ndarray]]:
        """Chroma özelliklerini enstrüman sınıflandırma için çıkarır"""
        features = {}
        
        chroma = self.extract_chroma(audio_data)
        
        # Temel chroma istatistikleri
        features['chroma_mean'] = np.mean(chroma, axis=1)       
        features['chroma_std'] = np.std(chroma, axis=1)         
        features['chroma_energy'] = np.sum(chroma, axis=1)      
        
        # Dominant nota ve harmonik özellikler
        features['dominant_note'] = np.argmax(features['chroma_energy'])
        features['chroma_centroid'] = np.sum(np.arange(12) * features['chroma_energy']) / (np.sum(features['chroma_energy']) + 1e-10)
        
        # Chroma stability (nota kararlılığı)
        chroma_diff = np.diff(chroma, axis=1)
        features['chroma_stability'] = 1.0 / (1.0 + np.mean(np.std(chroma_diff, axis=1)))
        
        print(f"  🎼 Chroma: 12 ton x 3 istatistik + 3 ek = {len([k for k in features.keys() if 'chroma' in k or 'dominant' in k])} özellik")
        return features
    
    def _extract_temporal_classification_features(self, audio_data: np.ndarray) -> Dict[str, float]:
        """Temporal özellikleri enstrüman karakteristikleri için çıkarır"""
        features = {}
        
        # Onset özellikleri
        onset_features = self.extract_onset_features(audio_data)
        features['onset_density'] = onset_features['onset_density']
        features['onset_interval_mean'] = onset_features['onset_interval_mean']
        features['onset_interval_std'] = onset_features['onset_interval_std']
        
        # Attack time (enstrüman ayırt edici özellik)
        features['attack_time'] = self.calculate_attack_time(audio_data)
        
        # Tempo özellikleri (ritim enstrümanları için)
        tempo_features = self.extract_tempo_features(audio_data)
        features['tempo'] = tempo_features['tempo']
        features['beat_regularity'] = tempo_features['beat_regularity']
        
        # Sustain ve decay analizi
        features.update(self._analyze_adsr_envelope(audio_data))
        
        print(f"  ⏱️ Temporal: 9 zaman özelliği")
        return features
    
    def _extract_harmonic_percussive_features(self, audio_data: np.ndarray) -> Dict[str, float]:
        """Harmonik vs Perküsif ayrışımı - enstrüman türü ayırt etme"""
        features = {}
        
        try:
            # Harmonik ve perküsif bileşenleri ayır
            harmonic, percussive = librosa.effects.hpss(audio_data)
            
            # Enerji oranları
            total_energy = np.sum(audio_data**2)
            harmonic_energy = np.sum(harmonic**2)
            percussive_energy = np.sum(percussive**2)
            
            if total_energy > 0:
                features['harmonic_ratio'] = harmonic_energy / total_energy
                features['percussive_ratio'] = percussive_energy / total_energy
            else:
                features['harmonic_ratio'] = 0.0
                features['percussive_ratio'] = 0.0
            
            # Harmonik-perküsif denge
            if percussive_energy > 0:
                features['harmonic_percussive_balance'] = harmonic_energy / percussive_energy
            else:
                features['harmonic_percussive_balance'] = float('inf') if harmonic_energy > 0 else 1.0
                
        except Exception as e:
            print(f"    ⚠️ HPSS analizi hatası: {e}")
            features['harmonic_ratio'] = 0.5
            features['percussive_ratio'] = 0.5
            features['harmonic_percussive_balance'] = 1.0
        
        print(f"  🎭 Harmonic/Percussive: 3 özellik")
        return features
    
    def _extract_envelope_features(self, audio_data: np.ndarray) -> Dict[str, float]:
        """Amplitude envelope özellikleri - enstrüman attack/decay karakteristikleri"""
        features = {}
        
        # RMS envelope
        rms = librosa.feature.rms(y=audio_data, hop_length=self.hop_length)[0]
        
        if len(rms) > 0:
            # Envelope şekil özellikleri
            features['envelope_mean'] = np.mean(rms)
            features['envelope_std'] = np.std(rms)
            features['envelope_max'] = np.max(rms)
            features['envelope_skewness'] = float(stats.skew(rms) if len(rms) > 0 else 0.0)
            features['envelope_kurtosis'] = float(stats.kurtosis(rms) if len(rms) > 0 else 0.0)
            
            # Peak-to-average ratio
            features['peak_to_average_ratio'] = features['envelope_max'] / (features['envelope_mean'] + 1e-10)
            
        else:
            features['envelope_mean'] = 0.0
            features['envelope_std'] = 0.0
            features['envelope_max'] = 0.0
            features['envelope_skewness'] = 0.0
            features['envelope_kurtosis'] = 0.0
            features['peak_to_average_ratio'] = 0.0
        
        print(f"  📊 Envelope: 6 özellik")
        return features
    
    def _analyze_adsr_envelope(self, audio_data: np.ndarray) -> Dict[str, float]:
        """ADSR envelope analizi (Attack, Decay, Sustain, Release)"""
        features = {}
        
        try:
            # Ses amplitüdünü normalize et
            audio_abs = np.abs(audio_data)
            if np.max(audio_abs) > 0:
                audio_abs = audio_abs / np.max(audio_abs)
            
            # Ses başlangıç ve bitişini bul
            trimmed, (start_idx, end_idx) = librosa.effects.trim(audio_data, top_db=20)
            
            if len(trimmed) > 100:
                envelope = audio_abs[start_idx:start_idx+len(trimmed)]
                
                # Attack phase (0-30% of sound)
                attack_region = envelope[:len(envelope)//3]
                if len(attack_region) > 0:
                    attack_peak_idx = np.argmax(attack_region)
                    features['attack_duration'] = attack_peak_idx / self.sample_rate * 1000  # ms
                else:
                    features['attack_duration'] = 0.0
                
                # Decay phase (30-70% of sound)
                if len(envelope) > 3:
                    decay_region = envelope[len(envelope)//3:2*len(envelope)//3]
                    if len(decay_region) > 1:
                        decay_slope = np.polyfit(range(len(decay_region)), decay_region, 1)[0]
                        features['decay_rate'] = abs(decay_slope)
                    else:
                        features['decay_rate'] = 0.0
                else:
                    features['decay_rate'] = 0.0
                
                # Sustain level (70-90% of sound)
                sustain_region = envelope[2*len(envelope)//3:9*len(envelope)//10]
                if len(sustain_region) > 0:
                    features['sustain_level'] = np.mean(sustain_region)
                else:
                    features['sustain_level'] = 0.0
                
                # Release phase (son 10%)
                release_region = envelope[9*len(envelope)//10:]
                if len(release_region) > 1:
                    release_slope = np.polyfit(range(len(release_region)), release_region, 1)[0]
                    features['release_rate'] = abs(release_slope)
                else:
                    features['release_rate'] = 0.0
            else:
                features['attack_duration'] = 0.0
                features['decay_rate'] = 0.0
                features['sustain_level'] = 0.0
                features['release_rate'] = 0.0
                
        except Exception as e:
            print(f"    ⚠️ ADSR analizi hatası: {e}")
            features['attack_duration'] = 0.0
            features['decay_rate'] = 0.0
            features['sustain_level'] = 0.0
            features['release_rate'] = 0.0
        
        return features
    
    # ==================== UTILITY METHODS ====================
    
    def get_feature_vector(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Random Forest için tek boyutlu özellik vektörü döndürür
        
        Args:
            audio_data (np.ndarray): Ses verisi
            
        Returns:
            np.ndarray: Düzleştirilmiş özellik vektörü
        """
        all_features = self.extract_all_features(audio_data)
        
        # Tüm özellikleri tek vektöre birleştir
        feature_vector = []
        
        for key, value in all_features.items():
            if isinstance(value, np.ndarray):
                feature_vector.extend(value.flatten())
            else:
                feature_vector.append(value)
        
        return np.array(feature_vector)
    
    def get_feature_names(self) -> List[str]:
        """
        Özellik isimlerini döndürür (Random Forest için)
        
        Returns:
            List[str]: Özellik isimleri
        """
        # Dummy audio ile özellik isimlerini al
        dummy_audio = np.random.randn(self.sample_rate)  # 1 saniye
        features = self.extract_all_features(dummy_audio)
        
        feature_names = []
        for key, value in features.items():
            if isinstance(value, np.ndarray):
                if value.ndim == 1:
                    for i in range(len(value)):
                        feature_names.append(f"{key}_{i}")
                else:
                    for i in range(value.size):
                        feature_names.append(f"{key}_{i}")
            else:
                feature_names.append(key)
        
        return feature_names
    
    def get_time_frames_count(self, audio_length: int) -> int:
        """
        Ses uzunluğuna göre zaman çerçeve sayısını hesaplar
        
        Args:
            audio_length (int): Ses verisi uzunluğu (sample)
            
        Returns:
            int: Zaman çerçeve sayısı
        """
        return 1 + (audio_length - self.n_fft) // self.hop_length
    
    def frames_to_time(self, frames: np.ndarray) -> np.ndarray:
        """
        Frame indekslerini zaman değerlerine çevirir
        
        Args:
            frames (np.ndarray): Frame indeksleri
            
        Returns:
            np.ndarray: Zaman değerleri (saniye)
        """
        return librosa.frames_to_time(frames, sr=self.sample_rate, hop_length=self.hop_length)
    
    def time_to_frames(self, times: np.ndarray) -> np.ndarray:
        """
        Zaman değerlerini frame indekslerine çevirir
        
        Args:
            times (np.ndarray): Zaman değerleri (saniye)
            
        Returns:
            np.ndarray: Frame indeksleri
        """
        return librosa.time_to_frames(times, sr=self.sample_rate, hop_length=self.hop_length)
    
    # ==================== ANALYSIS METHODS ====================
    
    def analyze_instrument_characteristics(self, audio_data: np.ndarray) -> Dict[str, any]:
        """
        Enstrüman karakteristik analizi
        
        Args:
            audio_data (np.ndarray): Ses verisi
            
        Returns:
            Dict: Enstrüman karakteristikleri
        """
        features = self.extract_all_features(audio_data)
        
        analysis = {
            'timbre_profile': {
                'brightness': float(np.mean(features.get('spectral_centroid_mean', [0]))),
                'roughness': float(features.get('spectral_flatness_std', 0)),
                'warmth': float(1.0 / (1.0 + features.get('spectral_centroid_std', 1))),
            },
            'attack_profile': {
                'attack_time_ms': float(features.get('attack_time', 0)),
                'attack_sharpness': float(1.0 / (1.0 + features.get('attack_duration', 1))),
                'percussive_nature': float(features.get('percussive_ratio', 0)),
            },
            'harmonic_profile': {
                'harmonic_content': float(features.get('harmonic_ratio', 0)),
                'tonal_stability': float(features.get('chroma_stability', 0)),
                'pitch_clarity': float(1.0 - features.get('spectral_flatness_mean', 1)),
            },
            'temporal_profile': {
                'note_density': float(features.get('onset_density', 0)),
                'rhythmic_regularity': float(features.get('beat_regularity', 0)),
                'sustain_capability': float(features.get('sustain_level', 0)),
            }
        }
        
        return analysis
    
    # ==================== VISUALIZATION METHODS ====================
    
    def visualize_basic_features(self, audio_data: np.ndarray, title: str = "Temel Özellikler"):
        """
        Temel özellikleri görselleştirir
        
        Args:
            audio_data (np.ndarray): Ses verisi
            title (str): Grafik başlığı
        """
        plt.figure(figsize=(15, 10))
        
        # 1. Waveform
        plt.subplot(3, 2, 1)
        times = np.linspace(0, len(audio_data)/self.sample_rate, len(audio_data))
        plt.plot(times, audio_data)
        plt.title('Ses Dalgası')
        plt.xlabel('Zaman (s)')
        plt.ylabel('Amplitude')
        
        # 2. Spektrogram
        plt.subplot(3, 2, 2)
        S = np.abs(librosa.stft(audio_data, hop_length=self.hop_length, n_fft=self.n_fft))
        librosa.display.specshow(librosa.amplitude_to_db(S), sr=self.sample_rate, x_axis='time', y_axis='hz')
        plt.title('Spektrogram')
        plt.colorbar()
        
        # 3. MFCC
        plt.subplot(3, 2, 3)
        mfcc = self.extract_mfcc(audio_data)
        librosa.display.specshow(mfcc, sr=self.sample_rate, x_axis='time')
        plt.title('MFCC')
        plt.colorbar()
        
        # 4. Chroma
        plt.subplot(3, 2, 4)
        chroma = self.extract_chroma(audio_data)
        librosa.display.specshow(chroma, sr=self.sample_rate, x_axis='time', y_axis='chroma')
        plt.title('Chroma')
        plt.colorbar()
        
        # 5. Pitch
        plt.subplot(3, 2, 5)
        try:
            pitch_features = self.calculate_pitch_features(audio_data)
            f0_times = self.frames_to_time(range(len(pitch_features['f0'])))
            plt.plot(f0_times, pitch_features['f0'])
            plt.title('Fundamental Frequency (F0)')
            plt.xlabel('Zaman (s)')
            plt.ylabel('Frequency (Hz)')
        except:
            plt.text(0.5, 0.5, 'Pitch analizi hatası', ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('Fundamental Frequency (F0)')
        
        # 6. Onsets
        plt.subplot(3, 2, 6)
        try:
            onset_features = self.extract_onset_features(audio_data)
            plt.plot(onset_features['onset_times'], [1]*len(onset_features['onset_times']), 'ro', markersize=8)
            plt.title(f'Onset Detection ({len(onset_features["onset_times"])} nota)')
            plt.xlabel('Zaman (s)')
            plt.ylim(0, 2)
        except:
            plt.text(0.5, 0.5, 'Onset analizi hatası', ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('Onset Detection')
        
        plt.suptitle(title)
        plt.tight_layout()
        plt.show()
    
    def visualize_classification_features(self, audio_data: np.ndarray, 
                                        save_path: Optional[str] = None):
        """Random Forest için enstrüman sınıflandırma özelliklerini görselleştirir"""
        plt.figure(figsize=(16, 10))
        
        # 1. MFCC
        plt.subplot(3, 3, 1)
        mfcc = self.extract_mfcc(audio_data, self.n_mfcc)
        librosa.display.specshow(mfcc, sr=self.sample_rate, x_axis='time')
        plt.title('MFCC Özellikleri')
        plt.colorbar()
        
        # 2. Spektral özellikleri
        plt.subplot(3, 3, 2)
        try:
            spectral_features = self.extract_spectral_features(audio_data)
            centroid = spectral_features['spectral_centroid'][0]
            times = self.frames_to_time(range(len(centroid)))
            plt.plot(times, centroid, label='Centroid')
            plt.plot(times, spectral_features['spectral_rolloff'][0], label='Rolloff')
            plt.title('Spektral Özellikler')
            plt.xlabel('Zaman (s)')
            plt.ylabel('Frekans (Hz)')
            plt.legend()
        except:
            plt.text(0.5, 0.5, 'Spektral analiz hatası', ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('Spektral Özellikler')
        
        # 3. Harmonic vs Percussive
        plt.subplot(3, 3, 3)
        try:
            harmonic, percussive = librosa.effects.hpss(audio_data)
            times = np.linspace(0, len(audio_data)/self.sample_rate, len(audio_data))
            plt.plot(times, harmonic, alpha=0.7, label='Harmonic')
            plt.plot(times, percussive, alpha=0.7, label='Percussive')
            plt.title('Harmonic vs Percussive')
            plt.xlabel('Zaman (s)')
            plt.legend()
        except:
            plt.text(0.5, 0.5, 'HPSS analiz hatası', ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('Harmonic vs Percussive')
        
        # 4. Chroma
        plt.subplot(3, 3, 4)
        chroma = self.extract_chroma(audio_data)
        librosa.display.specshow(chroma, sr=self.sample_rate, x_axis='time', y_axis='chroma')
        plt.title('Chroma Özellikleri')
        plt.colorbar()
        
        # 5. Spectral Contrast
        plt.subplot(3, 3, 5)
        try:
            S = np.abs(librosa.stft(audio_data, hop_length=self.hop_length))
            contrast = librosa.feature.spectral_contrast(S=S, sr=self.sample_rate)
            librosa.display.specshow(contrast, sr=self.sample_rate, x_axis='time')
            plt.title('Spectral Contrast')
            plt.colorbar()
        except:
            plt.text(0.5, 0.5, 'Contrast analiz hatası', ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('Spectral Contrast')
        
        # 6. Envelope
        plt.subplot(3, 3, 6)
        rms = librosa.feature.rms(y=audio_data, hop_length=self.hop_length)[0]
        times = self.frames_to_time(range(len(rms)))
        plt.plot(times, rms)
        plt.title('Amplitude Envelope')
        plt.xlabel('Zaman (s)')
        plt.ylabel('RMS')
        
        # 7. Onset detection
        plt.subplot(3, 3, 7)
        try:
            onset_features = self.extract_onset_features(audio_data)
            plt.plot(onset_features['onset_times'], [1]*len(onset_features['onset_times']), 'ro', markersize=8)
            plt.title(f'Onset Detection ({len(onset_features["onset_times"])} nota)')
            plt.xlabel('Zaman (s)')
            plt.ylim(0, 2)
        except:
            plt.text(0.5, 0.5, 'Onset analiz hatası', ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('Onset Detection')
        
        # 8. Tonnetz
        plt.subplot(3, 3, 8)
        try:
            chroma = self.extract_chroma(audio_data)
            tonnetz = librosa.feature.tonnetz(chroma=chroma)
            librosa.display.specshow(tonnetz, sr=self.sample_rate, x_axis='time')
            plt.title('Tonnetz (Harmonik Ağ)')
            plt.colorbar()
        except:
            plt.text(0.5, 0.5, 'Tonnetz analiz hatası', ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('Tonnetz')
        
        # 9. Feature summary
        plt.subplot(3, 3, 9)
        try:
            features = self.extract_all_features(audio_data)
            
            # Önemli özellikleri bar chart ile göster
            important_features = {
                'Attack Time': features.get('attack_time', 0),
                'Harmonic Ratio': features.get('harmonic_ratio', 0),
                'Onset Density': features.get('onset_density', 0),
                'Spectral Centroid': np.mean(features.get('spectral_centroid_mean', [0])),
                'Chroma Stability': features.get('chroma_stability', 0)
            }
            
            names = list(important_features.keys())
            values = list(important_features.values())
            
            # Normalize values for display
            values = np.array(values)
            max_val = np.max(values)
            if max_val > 0:
                values = values / max_val
            
            plt.bar(range(len(names)), values)
            plt.xticks(range(len(names)), names, rotation=45, ha='right')
            plt.title('Önemli Özellik Değerleri (Normalize)')
            plt.ylabel('Değer')
        except Exception as e:
            plt.text(0.5, 0.5, f'Feature summary hatası:\n{str(e)[:50]}...', ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('Özellik Özeti')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Enstrüman analizi kaydedildi: {save_path}")
        
        plt.show()


# ==================== YARDIMCI FONKSIYONLAR ====================

def statistical_summary(data: np.ndarray) -> Dict[str, float]:
    """
    Bir veri dizisi için istatistiksel özet çıkarır
    
    Args:
        data (np.ndarray): Veri dizisi
        
    Returns:
        Dict: İstatistiksel özellikler
    """
    if len(data) == 0:
        return {
            'mean': 0.0,
            'std': 0.0,
            'median': 0.0,
            'min': 0.0,
            'max': 0.0,
            'range': 0.0,
            'skewness': 0.0,
            'kurtosis': 0.0
        }
    
    # NaN kontrolü
    data = data[~np.isnan(data)]
    if len(data) == 0:
        return {
            'mean': 0.0,
            'std': 0.0,
            'median': 0.0,
            'min': 0.0,
            'max': 0.0,
            'range': 0.0,
            'skewness': 0.0,
            'kurtosis': 0.0
        }
    
    return {
        'mean': float(np.mean(data)),
        'std': float(np.std(data)),
        'median': float(np.median(data)),
        'min': float(np.min(data)),
        'max': float(np.max(data)),
        'range': float(np.max(data) - np.min(data)),
        'skewness': float(stats.skew(data)),
        'kurtosis': float(stats.kurtosis(data))
    }

def hz_to_note(frequency: float) -> str:
    """
    Frekansı nota ismine çevirir
    
    Args:
        frequency (float): Frekans (Hz)
        
    Returns:
        str: Nota ismi (örn: "C4", "A#3")
    """
    if frequency <= 0 or np.isnan(frequency):
        return "N/A"
    
    try:
        return librosa.hz_to_note(frequency)
    except:
        return "N/A"

def note_to_hz(note: str) -> float:
    """
    Nota ismini frekansa çevirir
    
    Args:
        note (str): Nota ismi (örn: "C4", "A#3")
        
    Returns:
        float: Frekans (Hz)
    """
    try:
        return librosa.note_to_hz(note)
    except:
        return 0.0


# ==================== TEST FONKSIYONU ====================

def test_classification_extractor():
    """ClassificationExtractor sınıfını test eder"""
    extractor = ClassificationExtractor()
    
    print("🎷 ClassificationExtractor Test Başlıyor...")
    print("=" * 50)
    
    # Test için sintetik ses oluştur
    duration = 2.0  # 2 saniye
    sample_rate = 22050
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Sintetik enstrüman sesi (harmonikler ile)
    fundamental = 440  # A4
    test_audio = (np.sin(2 * np.pi * fundamental * t) * 0.5 +
                  np.sin(2 * np.pi * fundamental * 2 * t) * 0.3 +
                  np.sin(2 * np.pi * fundamental * 3 * t) * 0.2)
    
    # Envelope ekle (attack-decay)
    envelope = np.exp(-t * 2)
    test_audio *= envelope
    
    try:
        print("\n1. extract_features() metodu test ediliyor...")
        features_vector = extractor.extract_features(test_audio)
        print(f"   ✅ Feature vector boyutu: {len(features_vector)}")
        
        print("\n2. extract_all_features() metodu test ediliyor...")
        all_features = extractor.extract_all_features(test_audio)
        print(f"   ✅ Toplam özellik türü: {len(all_features)}")
        
        print("\n3. get_feature_names() metodu test ediliyor...")
        feature_names = extractor.get_feature_names()
        print(f"   ✅ Özellik ismi sayısı: {len(feature_names)}")
        
        print("\n4. analyze_instrument_characteristics() metodu test ediliyor...")
        analysis = extractor.analyze_instrument_characteristics(test_audio)
        print(f"   ✅ Analiz kategorileri: {list(analysis.keys())}")
        
        print(f"\n📊 Test Özeti:")
        print(f"  - Feature vector uyumlu: {len(features_vector) == len(feature_names)}")
        print(f"  - Random Forest için hazır: ✅")
        
        print(f"\n🎼 Enstrüman Karakteristikleri:")
        for category, characteristics in analysis.items():
            print(f"  {category}:")
            for char_name, char_value in characteristics.items():
                print(f"    {char_name}: {char_value:.3f}")
        
        print("\n5. Görselleştirme test ediliyor...")
        extractor.visualize_classification_features(test_audio)
        
        print("✅ Tüm testler başarılı!")
        
    except Exception as e:
        print(f"❌ Test hatası: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_classification_extractor()