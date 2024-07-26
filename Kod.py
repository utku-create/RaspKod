import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import scipy.io.wavfile
import sounddevice as sd
import wavio
import speech_recognition as sr

# Ses tanıma fonksiyonu
def sesi_kaydet(filename):
    r = sr.Recognizer()

    with sr.AudioFile(filename) as kaynak:
        print("Ses dosyası yükleniyor...")
        ses = r.record(kaynak)

        söylenen_cümle = ""
        dil = "tr"  # Varsayılan dil Türkçe

        try:
            # Türkçe olarak tanımaya çalış
            söylenen_cümle = r.recognize_google(ses, language="tr-TR")
            print("Söylenen cümle (Türkçe):", söylenen_cümle)
            dil = "tr"
        except sr.UnknownValueError:
            pass  # Türkçe olarak anlayamadık
        except sr.RequestError:
            print("Üzgünüm, servisle ilgili bir problem var (Türkçe).")

        if not söylenen_cümle:
            try:
                # İngilizce olarak tanımaya çalış
                söylenen_cümle = r.recognize_google(ses, language="en-US")
                print("Söylenen cümle (İngilizce):", söylenen_cümle)
                dil = "en"
            except sr.UnknownValueError:
                pass  # İngilizce olarak anlayamadık
            except sr.RequestError:
                print("Üzgünüm, servisle ilgili bir problem var (İngilizce).")
            except Exception as e:
                print("Bir hata oluştu: " + str(e))

        if not söylenen_cümle:
            try:
                # Fransızca olarak tanımaya çalış
                söylenen_cümle = r.recognize_google(ses, language="fr-FR")
                print("Söylenen cümle (Fransızca):", söylenen_cümle)
                dil = "fr"
            except sr.UnknownValueError:
                pass  # Fransızca olarak anlayamadık
            except sr.RequestError:
                print("Üzgünüm, servisle ilgili bir problem var (Fransızca).")
            except Exception as e:
                print("Bir hata oluştu: " + str(e))

        if not söylenen_cümle:
            try:
                # İspanyolca olarak tanımaya çalış
                söylenen_cümle = r.recognize_google(ses, language="es-ES")
                print("Söylenen cümle (İspanyolca):", söylenen_cümle)
                dil = "es"
            except sr.UnknownValueError:
                pass  # İspanyolca olarak anlayamadık
            except sr.RequestError:
                print("Üzgünüm, servisle ilgili bir problem var (İspanyolca).")
            except Exception as e:
                print("Bir hata oluştu: " + str(e))

    return söylenen_cümle, dil

# Kontrol edilecek kelimelerin listesi
kelime_listesi = ["yardım", "destek", "imdat",
                  "acil", "kurtarın", "beni duyan var mı", "sesimi duyan var mı",
                  "yardıma ihtiyacım var", "hey", "help", "trapped", "emergency",
                  "rescue", "injured", "stuck", "save", "need", "aid", "urgent","Aide", "Urgent",
                  "Sauvez-moi", "Danger", "Pompier",
                  "Ambulance", "Aidez-moi", "S'il vous plaît", "Blessé", "Perdu","ayuda", "agua", "comida",
                  "medicina", "rescate", "refugio", "familia", "emergencia", "salud", "comunicación"]

# Ses kaydı parametreleri
duration = 5  # Kayıt süresi (saniye)
samplerate = 44100  # Örnekleme hızı (Hz)

while True:
    # Ses kaydını al
    print("Recording...")
    recording = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=2)
    sd.wait()  # Kayıt tamamlanana kadar bekle
    print("Recording complete")

    # Kaydı dosyaya yaz
    filename = "recording.wav"
    wavio.write(filename, recording, samplerate, sampwidth=3)  # 24-bit kayıt

    # Kelime tanıma işlemi
    yazı, dil = sesi_kaydet(filename)

    for kelime in kelime_listesi:
        if kelime.lower() in yazı.lower():
            print(f"'{kelime}' kelimesi tanındı!")


    # Çıkış kelimeleri kontrolü
    if "çıkış" in yazı.lower() or "exit" in yazı.lower():
        if dil == "tr":
            print("Program Türkçe olarak sonlandırılıyor.")
        elif dil == "en":
            print("Program İngilizce olarak sonlandırılıyor.")
        elif dil == "fr":
            print("Program Fransızca olarak sonlandırılıyor.")
        else:
            print("Program İspanyolca olarak sonlandırılıyor.")
        break
    else:
        # Librosa kullanarak ses dosyasını yükle
        librosa_audio, librosa_sample_rate = librosa.load(filename, sr=None)

        # Grafik oluşturma
        plt.figure(figsize=(15, 20))

        # Original Audio - 24BIT
        plt.subplot(5, 1, 1)
        plt.title('Original Audio - 24BIT')
        plt.plot(librosa_audio)

        # MFCC'leri çizdirin
        plt.subplot(5, 1, 2)
        mfccs = librosa.feature.mfcc(y=librosa_audio, sr=librosa_sample_rate, n_mfcc=40)
        librosa.display.specshow(mfccs, sr=librosa_sample_rate, x_axis='time', y_axis='hz')
        plt.title('Librosa MFCC Plot')
        plt.colorbar()
        print(mfccs.shape)

        # STFT kullanarak spektrogramı çizdirin
        plt.subplot(5, 1, 3)
        X = librosa.stft(librosa_audio)
        Xdb = librosa.amplitude_to_db(abs(X))
        librosa.display.specshow(Xdb, sr=librosa_sample_rate, x_axis='time', y_axis='hz')
        plt.title('Librosa STFT Plot')
        plt.colorbar(format='%+2.0f dB')

        # Mel spektrogram çizdirme
        plt.subplot(5, 1, 4)
        S = librosa.feature.melspectrogram(y=librosa_audio, sr=librosa_sample_rate, n_mels=128)
        S_db = librosa.power_to_db(S, ref=np.max)
        librosa.display.specshow(S_db, sr=librosa_sample_rate, x_axis='time', y_axis='mel')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Mel Spectrogram')

        # scipy.io.wavfile kullanarak ses dosyasını yükleyin
        samplerate, data = scipy.io.wavfile.read(filename)

        # Eğer veri iki kanallı (stereo) ise, tek kanala (mono) dönüştürün
        if len(data.shape) == 2 and data.shape[1] == 2:
            data = data.mean(axis=1)

        # spectrogram çizdirin
        plt.subplot(5, 1, 5)
        plt.specgram(data, Fs=samplerate)
        plt.title('Scipy Spectrogram Plot')

        plt.tight_layout()
        plt.show()


import os
import pyaudio
from google.cloud import speech
import queue

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "sa_speech_demo.json"

RATE = 16000
CHUNK = int(RATE / 10)

keywords_tr = ["yardım", "destek", "imdat", "acil", "kurtarın", "beni duyan var mı",
               "sesimi duyan var mı", "yardıma ihtiyacım var", "hey", "çıkış"]

class MicrophoneStream:
    def __init__(self, rate, chunk):
        self.rate = rate
        self.chunk = chunk
        self.buff = queue.Queue()
        self.closed = True

    def __enter__(self):
        self.audio_interface = pyaudio.PyAudio()
        self.audio_stream = self.audio_interface.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.rate,
            input=True,
            frames_per_buffer=self.chunk,
            stream_callback=self._fill_buffer,
        )
        self.closed = False
        return self

    def __exit__(self, type, value, traceback):
        self.audio_stream.stop_stream()
        self.audio_stream.close()
        self.closed = True
        self.buff.put(None)
        self.audio_interface.terminate()

    def _fill_buffer(self, in_data, frame_count, time_info, status_flags):
        self.buff.put(in_data)
        return None, pyaudio.paContinue

    def generator(self):
        while not self.closed:
            chunk = self.buff.get()
            if chunk is None:
                return
            data = [chunk]

            while True:
                try:
                    chunk = self.buff.get(block=False)
                    if chunk is None:
                        return
                    data.append(chunk)
                except queue.Empty:
                    break

            yield b"".join(data)

def listen_print_loop(responses, file_path, stream):
    with open(file_path, 'w') as f:
        for response in responses:
            if not response.results:
                continue

            result = response.results[0]
            if not result.alternatives:
                continue

            transcript = result.alternatives[0].transcript.lower()
            print(f'Transcript: {transcript}')
            f.write(f'Transcript: {transcript}\n')

            for keyword in keywords_tr:
                if keyword in transcript:
                    print(f'Kelime tanındı: {keyword}')
                    f.write(f'Kelime tanındı: {keyword}\n')
                    if keyword == "çıkış":
                        stream.closed = True
                        return

def transcribe_tr():
    client = speech.SpeechClient()
    file_path = 'transcript_tr.txt'

    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=RATE,
        language_code="tr-TR",
    )

    streaming_config = speech.StreamingRecognitionConfig(
        config=config,
        interim_results=True,
    )

    with MicrophoneStream(RATE, CHUNK) as stream:
        audio_generator = stream.generator()
        requests = (speech.StreamingRecognizeRequest(audio_content=content)
                    for content in audio_generator)

        responses = client.streaming_recognize(streaming_config, requests)

        listen_print_loop(responses, file_path, stream)

if __name__ == "__main__":
    transcribe_tr()


