import sounddevice as sd
from scipy.io.wavfile import write

def record_audio(duration, sample_rate, channels):
    print("Recording...")
    audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=channels)
    sd.wait()
    file_name = "recorded_audio.wav"
    write(file_name, sample_rate, audio_data)
    print("done")
    return file_name
