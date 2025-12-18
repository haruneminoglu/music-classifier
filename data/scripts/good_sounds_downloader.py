import os
import json
from pathlib import Path
from typing import Dict, List, Optional
import time
import shutil
import requests
import sys
import urllib.request
import zipfile
import tarfile


class DatasetUtils:
    """
    Veri seti indirme ve organize etme için ortak fonksiyonlar
    """
    
    @staticmethod
    def create_directory_structure(base_path: Path, subdirs: List[str]) -> None:
        """Klasör yapısını oluşturur"""
        base_path.mkdir(parents=True, exist_ok=True)
        for subdir in subdirs:
            (base_path / subdir).mkdir(parents=True, exist_ok=True)
    
    @staticmethod
    def download_file_with_progress(url: str, filepath: Path, show_progress: bool = True) -> bool:
        """Dosyayı progress bar ile indirir"""
        print(f"📥 İndiriliyor: {filepath.name}")
        
        def progress_hook(block_num, block_size, total_size):
            if show_progress and total_size > 0:
                downloaded = block_num * block_size
                percent = min(100, (downloaded * 100) / total_size)
                print(f"\r  Progress: {percent:.1f}%", end="", flush=True)
        
        try:
            filepath.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(url, filepath, progress_hook)
            if show_progress:
                print(f"\n✅ İndirildi: {filepath}")
            return True
        except Exception as e:
            print(f"\n❌ İndirme hatası: {e}")
            return False
    
    @staticmethod
    def extract_archive(archive_path: Path, extract_to: Path, archive_type: str = "auto") -> bool:
        """Arşiv dosyasını çıkarır (zip, tar, tar.gz destekli)"""
        print(f"📦 Çıkarılıyor: {archive_path.name}")
        
        try:
            extract_to.mkdir(parents=True, exist_ok=True)
            
            # Otomatik tip algılama
            if archive_type == "auto":
                if archive_path.suffix.lower() == '.zip':
                    archive_type = "zip"
                elif archive_path.name.endswith('.tar.gz') or archive_path.suffix.lower() == '.tgz':
                    archive_type = "tar.gz"
                elif archive_path.suffix.lower() == '.tar':
                    archive_type = "tar"
            
            # Çıkarma işlemi
            if archive_type == "zip":
                with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_to)
            elif archive_type in ["tar", "tar.gz"]:
                mode = "r:gz" if archive_type == "tar.gz" else "r"
                with tarfile.open(archive_path, mode) as tar_ref:
                    tar_ref.extractall(extract_to)
            else:
                raise ValueError(f"Desteklenmeyen arşiv tipi: {archive_type}")
            
            print(f"✅ Çıkarıldı: {extract_to}")
            return True
        except Exception as e:
            print(f"❌ Çıkarma hatası: {e}")
            return False
    
    @staticmethod
    def copy_files_with_rename(source_dir: Path, target_dir: Path, 
                             file_pattern: str, prefix: str) -> int:
        """Dosyaları yeni isimlerle kopyalar"""
        target_dir.mkdir(parents=True, exist_ok=True)
        
        files = list(source_dir.glob(file_pattern))
        for i, file_path in enumerate(files):
            new_name = f"{prefix}_{i+1:03d}{file_path.suffix}"
            target_path = target_dir / new_name
            shutil.copy2(file_path, target_path)
        
        return len(files)
    
    @staticmethod
    def cleanup_temp_files(paths: List[Path]) -> None:
        """Geçici dosyaları temizler"""
        for path in paths:
            try:
                if path.is_file():
                    path.unlink()
                elif path.is_dir():
                    shutil.rmtree(path)
            except Exception as e:
                print(f"⚠️ Temizleme hatası {path}: {e}")
    
    @staticmethod
    def verify_audio_files(directory: Path, expected_extensions: List[str] = ['.wav', '.mp3']) -> Dict[str, int]:
        """Ses dosyalarını doğrular"""
        verification = {}
        
        for ext in expected_extensions:
            files = list(directory.rglob(f"*{ext}"))
            verification[ext] = len(files)
        
        return verification
    
    @staticmethod
    def save_dataset_info(info_path: Path, dataset_info: Dict) -> None:
        """Dataset bilgilerini JSON olarak kaydeder"""
        info_path.parent.mkdir(parents=True, exist_ok=True)
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(dataset_info, f, indent=2, ensure_ascii=False)
    
    @staticmethod
    def load_dataset_info(info_path: Path) -> Optional[Dict]:
        """Dataset bilgilerini JSON'dan yükler"""
        if info_path.exists():
            with open(info_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return None
    
    @staticmethod
    def print_summary(title: str, file_counts: Dict[str, int], 
                     dataset_path: Path, success: bool = True) -> None:
        """Kurulum özetini yazdırır"""
        print(f"\n📊 {title}")
        print("-" * 50)
        print(f"📁 Dataset konumu: {dataset_path}")
        
        total_files = sum(file_counts.values())
        for category, count in file_counts.items():
            if count > 0:
                print(f"  {category}: {count} dosya")
        
        print(f"  TOPLAM: {total_files} dosya")
        
        if success:
            print("✅ Kurulum başarıyla tamamlandı!")
        else:
            print("⚠️ Kurulumda bazı sorunlar oluştu")
    
    @staticmethod
    def get_project_root() -> Path:
        """Proje kök dizinini bulur"""
        current = Path(__file__).resolve()
        # good_sounds_downloader.py dosyasının konumuna göre ayarla
        # Eğer scripts/classification/ içindeyse:
        if 'scripts' in current.parts:
            # scripts/classification/good_sounds_downloader.py -> proje kökü
            return current.parent.parent.parent
        else:
            # Doğrudan proje kökündeyse
            return current.parent
    
    @staticmethod
    def ensure_data_structure(project_root: Path) -> Path:
        """Standart data klasör yapısını oluşturur"""
        data_dir = project_root / "data"
        raw_audio_dir = data_dir / "raw_audio" / "classification"
        processed_dir = data_dir / "processed"
        
        data_dir.mkdir(exist_ok=True)
        raw_audio_dir.mkdir(parents=True, exist_ok=True)
        processed_dir.mkdir(exist_ok=True)
        
        return data_dir


class InstrumentMapper:
    """Enstrüman eşleme ve kategorizasyon için yardımcı sınıf"""
    
    # 5 hedef enstrüman
    TARGET_INSTRUMENTS = ['violin', 'flute', 'cello', 'trumpet', 'clarinet']
    
    # Standart enstrüman kategorileri
    INSTRUMENT_CATEGORIES = {
        'strings': ['violin', 'cello'],
        'woodwinds': ['flute', 'clarinet'],
        'brass': ['trumpet']
    }
    
    # Good-Sounds dataset instrument mapping (gerçek Good-Sounds isimleri)
    GOOD_SOUNDS_INSTRUMENTS = {
        'violin': 'violin',
        'flute': 'flute', 
        'cello': 'cello',
        'trumpet': 'trumpet',
        'clarinet': 'clarinet',
        # Good-Sounds'ta bu isimler tam olarak mevcut
    }
    
    @classmethod
    def get_category(cls, instrument: str) -> str:
        """Enstrümanın kategorisini döndürür"""
        for category, instruments in cls.INSTRUMENT_CATEGORIES.items():
            if instrument in instruments:
                return category
        return 'other'
    
    @classmethod
    def normalize_instrument_name(cls, name: str) -> str:
        """Enstrüman adını normalize eder"""
        name = name.lower().strip()
        # Bazı ortak değişimleri normalize et
        normalizations = {
            'electric_violin': 'violin',
            'acoustic_violin': 'violin',
            'concert_flute': 'flute',
            'transverse_flute': 'flute',
            'violoncello': 'cello',
            'double_bass': 'cello',  # Alternatif string enstrümanı
            'bb_clarinet': 'clarinet',
            'bass_clarinet': 'clarinet',
            'piccolo_trumpet': 'trumpet',
            'cornet': 'trumpet',
        }
        return normalizations.get(name, name)


class GoodSoundsDownloader:
    """
    Good-Sounds dataset'ini indirir ve organize eder
    Streamlined version - sadece HTTP stream download ile
    """
    
    def __init__(self, project_root: Path = None):
        # Proje kök dizinini belirle
        if project_root is None:
            self.project_root = DatasetUtils.get_project_root()
        else:
            self.project_root = Path(project_root)
            
        if self.project_root.name == 'data':
            self.project_root = self.project_root.parent
            
        # Data yapısını oluştur
        DatasetUtils.ensure_data_structure(self.project_root)
        self.data_dir = self.project_root / "data"
        self.raw_audio_dir = self.data_dir / "raw_audio" / "classification"
        self.good_sounds_dir = self.raw_audio_dir / "good_sounds"
        
        # Zenodo record ID
        self.zenodo_record_id = "820937"
        
        # Download URL'leri (öncelik sırasıyla)
        self.download_urls = [
            f"https://zenodo.org/api/records/{self.zenodo_record_id}/files/good-sounds.zip/content",
            f"https://zenodo.org/record/{self.zenodo_record_id}/files/good-sounds.zip?download=1",
            f"https://zenodo.org/records/{self.zenodo_record_id}/files/good-sounds.zip?download=1",
            f"https://zenodo.org/api/files/7dca1bfd-9d64-41f5-b0b8-e50bb8a45fb4/good-sounds.zip"
        ]
        
        # Hedef enstrümanlar
        self.target_instruments = InstrumentMapper.GOOD_SOUNDS_INSTRUMENTS
        
        print("🎵 Good-Sounds Downloader hazır (Streamlined version)")
        print(f"📁 Proje kökü: {self.project_root}")
        print(f"📁 Data klasörü: {self.data_dir}")
        print(f"📁 Good-Sounds hedef klasör: {self.good_sounds_dir}")
    
    def download_with_requests(self, url: str, filepath: Path) -> bool:
        """Requests ile streaming download - geliştirilmiş versiyon"""
        try:
            print(f"📥 İndiriliyor: {url}")
            
            # Session kullanarak daha stabil bağlantı
            session = requests.Session()
            session.headers.update({
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            })
            
            with session.get(url, stream=True, timeout=60) as response:
                response.raise_for_status()  # HTTP hatalarını otomatik kontrol et
                
                filepath.parent.mkdir(parents=True, exist_ok=True)
                
                total_size = int(response.headers.get('content-length', 0))
                downloaded_size = 0
                chunk_size = 8192 * 16  # Daha büyük chunk size (128KB)
                
                print(f"  📊 Dosya boyutu: {total_size / (1024**3):.1f} GB" if total_size > 0 else "  📊 Dosya boyutu: Bilinmiyor")
                
                with open(filepath, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)
                            downloaded_size += len(chunk)
                            
                            if total_size > 0:
                                progress = (downloaded_size * 100) / total_size
                                downloaded_gb = downloaded_size / (1024**3)
                                print(f"\r  📥 Progress: {progress:.1f}% ({downloaded_gb:.1f} GB)", end="", flush=True)
                
                print(f"\n✅ İndirme tamamlandı: {filepath.name}")
                
                # Dosya boyutu kontrolü
                if filepath.stat().st_size < 1024 * 1024:  # 1MB'den küçükse problem var
                    print("❌ İndirilen dosya çok küçük, muhtemelen hatalı")
                    return False
                
                return True
                
        except requests.exceptions.RequestException as e:
            print(f"\n❌ HTTP hatası: {e}")
            return False
        except Exception as e:
            print(f"\n❌ İndirme hatası: {e}")
            return False
    
    def download_dataset(self, temp_dir: Path) -> Path:
        """Dataset'i indir - sadece HTTP stream ile"""
        good_sounds_archive = temp_dir / "good-sounds.zip"
        
        print("🔄 Good-Sounds dataset indiriliyor...")
        
        # Farklı URL'leri sırasıyla dene
        for i, url in enumerate(self.download_urls, 1):
            print(f"\n🔄 Deneme {i}/{len(self.download_urls)}")
            
            if self.download_with_requests(url, good_sounds_archive):
                return good_sounds_archive
            
            print(f"❌ URL {i} başarısız, sonraki deneniyor...")
            
            # Başarısız dosyayı temizle
            if good_sounds_archive.exists():
                good_sounds_archive.unlink()
        
        # Tüm URL'ler başarısız
        print("\n❌ Tüm indirme URL'leri başarısız!")
        print("\n📝 MANUEL İNDİRME TALİMATLARI:")
        print("1. Web tarayıcınızla şu adresi açın:")
        print("   https://zenodo.org/record/820937")
        print("2. 'good-sounds.zip' dosyasını bulun ve indirin (13.9 GB)")
        print(f"3. İndirilen dosyayı şu konuma taşıyın:")
        print(f"   {good_sounds_archive}")
        print("4. Script'i tekrar çalıştırın")
        
        # Kullanıcı manuel indirme yapana kadar bekle
        input("\n⏳ Manuel indirme tamamlandığında ENTER'e basın...")
        
        if good_sounds_archive.exists():
            print("✅ Manuel indirilen dosya bulundu!")
            return good_sounds_archive
        else:
            print("❌ Dosya bulunamadı!")
            return None
    
    def organize_good_sounds_files(self, source_dir: Path) -> Dict[str, int]:
        """Good-Sounds dosyalarını organize eder - sınırsız dosya"""
        print("🗂️ Good-Sounds dosyaları organize ediliyor...")
        
        # Good-Sounds ana klasörünü oluştur
        self.good_sounds_dir.mkdir(parents=True, exist_ok=True)
        
        # Her enstrüman için alt klasör oluştur
        instrument_dirs = list(self.target_instruments.values())
        DatasetUtils.create_directory_structure(self.good_sounds_dir, instrument_dirs)
        
        file_counts = {instrument: 0 for instrument in self.target_instruments.values()}
        
        # Audio dosyaları bul (.flac ve .wav)
        audio_files = []
        for ext in ['.flac', '.wav']:
            audio_files.extend(list(source_dir.rglob(f"*{ext}")))
        
        print(f"  📊 Toplam {len(audio_files)} audio dosyası bulundu")
        
        # Debug: Bulunan dosyalardan örnekler göster
        if audio_files:
            print("  📋 Örnek dosya adları:")
            for i, file in enumerate(audio_files[:5]):
                print(f"    {file.name}")
        
        # Her enstrüman için dosyaları filtrele ve organize et
        for source_name, target_instrument in self.target_instruments.items():
            target_dir = self.good_sounds_dir / target_instrument
            
            # Enstrüman adını normalize et
            normalized_name = InstrumentMapper.normalize_instrument_name(source_name)
            search_patterns = [source_name.lower(), normalized_name]
            
            # Enstrüman adına göre dosyaları filtrele
            instrument_files = []
            for audio_file in audio_files:
                file_lower = audio_file.name.lower()
                path_lower = str(audio_file).lower()
                
                # Daha kapsamlı pattern matching
                if any(pattern in file_lower or pattern in path_lower 
                       for pattern in search_patterns):
                    instrument_files.append(audio_file)
            
            if not instrument_files:
                print(f"  ⚠️ {source_name} -> {target_instrument}: Audio dosyası bulunamadı")
                continue
            
            # TÜM dosyaları kopyala (sınırlama yok)
            copied_count = 0
            for i, audio_file in enumerate(instrument_files):
                try:
                    # Yeni dosya adı (tutarlı format - .wav olarak kaydet)
                    new_name = f"{target_instrument}_{i+1:04d}.wav"  # 4 haneli padding
                    target_path = target_dir / new_name
                    
                    # Dosyayı kopyala
                    shutil.copy2(audio_file, target_path)
                    copied_count += 1
                    
                    # Progress göstergesi (her 100 dosyada bir)
                    if copied_count % 100 == 0:
                        print(f"    📥 {target_instrument}: {copied_count} dosya kopyalandı...")
                        
                except Exception as e:
                    print(f"    ⚠️ Kopyalama hatası {audio_file.name}: {e}")
            
            file_counts[target_instrument] = copied_count
            print(f"  ✅ {source_name} -> {target_instrument}: {copied_count} dosya (TÜM DOSYALAR)")
        
        return file_counts
    
    def download_and_setup_good_sounds(self) -> bool:
        """Good-Sounds dataset'ini indirir ve kurar"""
        print("🎵 GOOD-SOUNDS DATASET KURULUMU BAŞLIYOR")
        print("=" * 50)
        
        temp_dir = self.data_dir / "temp_good_sounds"
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Dataset'i indir
        archive_path = self.download_dataset(temp_dir)
        if not archive_path or not archive_path.exists():
            print("❌ İndirme başarısız!")
            return False
        
        # 2. Arşivi çıkar
        extract_dir = temp_dir / "extracted"
        success = DatasetUtils.extract_archive(archive_path, extract_dir, "zip")
        if not success:
            return False
        
        # 3. Çıkarılan dosyaların içindeki yapıyı analiz et
        print("🔍 Dataset yapısı analiz ediliyor...")
        
        # Mümkün kaynak dizinler
        potential_dirs = [
            extract_dir,
            extract_dir / "good-sounds",
            extract_dir / "Good-Sounds"
        ]
        
        source_dir = None
        for potential_dir in potential_dirs:
            if potential_dir.exists():
                audio_files = list(potential_dir.rglob("*.flac")) + list(potential_dir.rglob("*.wav"))
                if audio_files:
                    source_dir = potential_dir
                    print(f"  ✅ Dataset bulundu: {source_dir.name} ({len(audio_files)} audio dosyası)")
                    break
        
        if not source_dir:
            print("❌ Dataset içeriği bulunamadı!")
            return False
        
        # 4. Dosyaları organize et
        file_counts = self.organize_good_sounds_files(source_dir)
        
        # 5. Dataset bilgilerini kaydet
        dataset_info = {
            'dataset': 'Good-Sounds',
            'version': '1.0',
            'download_date': time.strftime('%Y-%m-%d %H:%M:%S'),
            'zenodo_record_id': self.zenodo_record_id,
            'instruments': file_counts,
            'total_files': sum(file_counts.values()),
            'source_info': {
                'original_format': '.flac',
                'converted_format': '.wav',
                'sample_rate': '48kHz',
                'bit_depth': '32-bit',
                'channels': 'mono',
                'total_size': '13.9 GB',
                'file_limit': 'No limit - all files copied'
            },
            'target_instruments': self.target_instruments
        }
        
        info_path = self.good_sounds_dir / "dataset_info.json"
        DatasetUtils.save_dataset_info(info_path, dataset_info)
        
        # 6. Geçici dosyaları temizle
        DatasetUtils.cleanup_temp_files([temp_dir])
        print("🧹 Geçici dosyalar temizlendi")
        
        # 7. Sonuç raporu
        total_files = sum(file_counts.values())
        success = total_files >= 10  # Minimum dosya beklentisi
        
        DatasetUtils.print_summary("GOOD-SOUNDS KURULUM TAMAMLANDI!", 
                                 file_counts, self.good_sounds_dir, success)
        
        return success
    
    def verify_installation(self) -> Dict[str, any]:
        """Kurulumu doğrular"""
        print("🔍 Good-Sounds kurulumu doğrulanıyor...")
        print(f"📁 Kontrol edilen klasör: {self.good_sounds_dir}")
        
        verification = {
            'success': True,
            'instruments': {},
            'total_files': 0,
            'total_duration': 0,
            'dataset_path': str(self.good_sounds_dir)
        }
        
        for instrument in self.target_instruments.values():
            instrument_dir = self.good_sounds_dir / instrument
            
            if instrument_dir.exists():
                wav_files = list(instrument_dir.glob("*.wav"))
                verification['instruments'][instrument] = len(wav_files)
                verification['total_files'] += len(wav_files)
                
                print(f"  ✅ {instrument}: {len(wav_files)} dosya")
            else:
                verification['instruments'][instrument] = 0
                verification['success'] = False
                print(f"  ❌ {instrument}: Klasör bulunamadı")
        
        if verification['total_files'] < 5:
            verification['success'] = False
            print(f"  ⚠️ Toplam dosya sayısı beklenenden az: {verification['total_files']}")
        
        return verification
    
    def get_dataset_statistics(self) -> Dict[str, any]:
        """Dataset istatistiklerini döndürür"""
        stats = {
            'dataset_name': 'Good-Sounds',
            'instruments': {},
            'categories': {}
        }
        
        for instrument in self.target_instruments.values():
            instrument_dir = self.good_sounds_dir / instrument
            if instrument_dir.exists():
                file_count = len(list(instrument_dir.glob("*.wav")))
                stats['instruments'][instrument] = file_count
                
                # Kategori bilgisi
                category = InstrumentMapper.get_category(instrument)
                if category not in stats['categories']:
                    stats['categories'][category] = 0
                stats['categories'][category] += file_count
        
        stats['total_files'] = sum(stats['instruments'].values())
        return stats


def main():
    """Ana çalıştırma fonksiyonu"""
    print("🎵 GOOD-SOUNDS DATASET DOWNLOADER (STREAMLINED VERSION)")
    print("=" * 60)
    
    downloader = GoodSoundsDownloader()
    
    # Dataset'i indir ve kur
    success = downloader.download_and_setup_good_sounds()
    
    if success:
        # Kurulumu doğrula
        verification = downloader.verify_installation()
        
        if verification['success']:
            # İstatistikleri göster
            stats = downloader.get_dataset_statistics()
            
            print(f"\n🎯 Kurulum Başarılı!")
            print(f"📁 Dataset konumu: {verification['dataset_path']}")
            print(f"📊 Toplam dosya: {verification['total_files']}")
            print(f"🎼 Enstrüman sayısı: {len(stats['instruments'])}")
            
            # Kategori dağılımını göster
            print(f"\n📂 Kategori dağılımı:")
            for category, count in stats['categories'].items():
                print(f"   {category}: {count} dosya")
            
            # Fine-tuning için hazır olduğunu belirt
            print(f"\n🚀 FINE-TUNING İÇİN HAZIR!")
            print(f"   📊 Enstrüman başına ortalama: {verification['total_files'] // len(stats['instruments'])} dosya")
            print(f"   🎯 Bu sayılar fine-tuning için yeterli olmalı")
            
            # Sonraki adım önerisi
            print(f"\n🔥 SONRAKI ADIM:")
            print(f"   cd {downloader.project_root}")
            print(f"   python data/scripts/preprocessing/data_manager.py")
            print(f"   # Veri setini hazırlamak için")
        else:
            print("\n⚠️ Doğrulama başarısız!")
    else:
        print("\n❌ Kurulum başarısız!")


if __name__ == "__main__":
    main()