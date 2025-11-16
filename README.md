# Nero

Lokalny asystent głosowy inspirowany Neuro-samą, zaprojektowany dla środowiska Windows 11 + Python 3.11 (64-bit) uruchamianego w wirtualnym środowisku `.venv`. Nero nasłuchuje mikrofonu, wykorzystuje detekcję mowy (VAD), transkrybuje wypowiedzi przez Faster-Whisper na GPU, generuje odpowiedzi lokalnym modelem llama.cpp (z fallbackiem OpenAI) i odtwarza je przez Piper (z fallbackiem OpenAI TTS). Architektura jest modułowa i przystosowana do sprzętu z RTX 4080 SUPER, 7800X3D i 32 GB RAM.

## Funkcje

- **Ciągłe nasłuchiwanie** – `AudioListener` przechwytuje strumień audio i pozwala wybrać urządzenie wejściowe.
- **VAD + segmentacja** – `VoiceActivityDetector` bazuje na `webrtcvad`, a dodatkowe progi (min. 700 ms) i profile długości „command”/„conversation” pilnują, by do ASR trafiały pełne wypowiedzi.
- **ASR OpenAI/Hybrid** – `ASRManager` potrafi przełączać między chmurowym `whisper-large-v3-turbo` a lokalnym Faster-Whisper (fallback przy błędach/limitach), wraz z licznikami kosztów.
- **Pamięć** – `MemoryManager` zapisuje rozmowy w SQLite + wektorach z Sentence-Transformers, żeby Nero pamiętał cele użytkownika.
- **LLM lokalny + chmurowy** – `LocalLLM` (llama-cpp-python) generuje odpowiedzi wg system promptu z `docs/system_prompt.txt`, `CloudLLM` służy jako fallback lub tryb hybrydowy z heurystyką intentów i limitami calli.
- **TTS z wyborem wyjścia audio** – `PiperTTS` steruje urządzeniem wyjściowym, a `CloudTTS` (OpenAI) jest rezerwą.
- **Profile pracy i override głosem** – jedno pole `profile` ustawia kombinację ASR/LLM (offline/balanced/cloud), a komendy typu „use cloud for this”/„local only for now” pozwalają tymczasowo wymusić backend.
- **Statystyki sesji** – Orchestrator zlicza sekundy audio wysłane do chmury, liczbę wywołań GPT i fallbacków, co ułatwia kontrolę kosztów.
- **Avatar GIF i tray UI (hook)** – konfiguracja `ui.avatar_gif` pozwala wskazać animację dla przyszłego komponentu w zasobniku systemowym.

## Wymagania sprzętowe i środowisko

- Windows 11 64-bit, Python 3.11 (oficjalny instalator x64).
- GPU NVIDIA RTX 4080 SUPER z aktualnymi sterownikami i CUDA (dla Faster-Whisper i llama.cpp).
- Procesor AMD Ryzen 7 7800X3D, 32 GB RAM lub więcej.
- Mikrofon oraz głośniki/słuchawki widoczne jako urządzenia w `sounddevice`.
- Notepad++ i CMD jako narzędzia pracy (np. edycja i polecenia w terminalu).

## Instalacja (CMD)

```cmd
cd C:\Users\TwojUzytkownik\Projects\Nero
python -m venv .venv
.\.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

Uzupełnij modele w katalogach z konfiguracji (np. `C:\\Models\\faster-whisper-large-v3`).

## Konfiguracja

Plik `config\config.json` zawiera wszystkie parametry działania:

- `profile` – szybkie przełączanie presetów (`offline`, `balanced`, `cloud-boost`, `full-cloud`).
- `audio` – częstotliwość próbkowania, rozmiar ramki, identyfikatory urządzeń oraz ustawienia segmentów (`min_segment_ms`, `segment_modes`).
- `vad` – agresywność i progi ciszy.
- `asr` – wybór providera (`local` / `openai` / `hybrid`), parametry modelu `whisper-large-v3-turbo`, limity (np. `max_monthly_seconds`) i sekcja `local` z Faster-Whisper.
- `llm` – tryb (`local`, `cloud`, `hybrid`), limity calli (`max_cloud_calls_per_*`), heurystyki hybrydowe oraz ustawienia llama.cpp.
- `cloud` – parametry OpenAI (model, zmienna z kluczem API) współdzielone przez LLM/ASR/TTS.
- `tts` – wybór między Piper a chmurą oraz konfiguracja urządzenia wyjściowego.
- `memory` – ścieżka bazy SQLite i model embeddingów.
- `ui` – ścieżka do animowanego avatara GIF wykorzystywanego przez przyszły tray icon.

> `config_loader.py` automatycznie wstrzykuje osobowość z `docs\system_prompt.txt`, więc wystarczy utrzymywać ją w jednym miejscu.

### Wybór urządzeń audio

Dostępne urządzenia możesz wypisać w Pythonie:

```python
from nero.audio_listener import AudioListener
for device in AudioListener.list_devices():
    print(device)
```

Skopiuj `index` lub `name` do `audio.input_device` (mikrofon) oraz `audio.output_device` (głośniki/wyjście TTS).

## Uruchamianie Nero

Przykładowy skrypt startowy (np. `run_nero.py`):

```python
import logging
from nero.config_loader import load_config
from nero.core.orchestrator import NeroOrchestrator

logging.basicConfig(level=logging.INFO)
config, persona = load_config()
orchestrator = NeroOrchestrator(config, persona)
try:
    orchestrator.run_forever()
except KeyboardInterrupt:
    orchestrator.shutdown()
```

Uruchom w CMD:

```cmd
.\.venv\Scripts\activate
python run_nero.py
```

## Architektura katalogów

```
nero
├─ config\config.json
├─ docs\system_prompt.txt
├─ requirements.txt
└─ src\nero
   ├─ asr_manager.py
   ├─ audio_listener.py
   ├─ vad.py
   ├─ whisper_asr.py
   ├─ local_llm.py
   ├─ cloud_llm.py
   ├─ memory_manager.py
   ├─ piper_tts.py
   ├─ cloud_tts.py
   ├─ config_loader.py
   └─ core\orchestrator.py
```

Każdy moduł posiada jednoznaczne API (np. `WhisperASR.transcribe()` zwraca `{"text": str, "language": str}`), dzięki czemu łatwo rozszerzać projekt – choćby o tray icon, dodatkowe efekty wizualne lub UI z wyświetleniem avatara GIF.