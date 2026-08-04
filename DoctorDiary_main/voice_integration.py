import os
import json
import pyaudio
import threading
import time
from google.cloud import speech_v2 as speech
from google.cloud.speech_v2.types import cloud_speech
from openai import OpenAI

# Configuration
RATE = 16000
CHUNK = 4096
LANGUAGES = ["en-IN", "hi-IN", "mr-IN"]
EXTRACTION_INTERVAL = 10  # seconds
MIN_EXTRACTION_INTERVAL = 2  # seconds

class VoiceIntegration:
    def __init__(self, project_id, credentials_path, openai_key):
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path
        os.environ["OPENAI_API_KEY"] = openai_key
        
        self.project_id = project_id
        self.speech_client = speech.SpeechClient()
        self.openai_client = OpenAI()
        self.recognizer = f"projects/{project_id}/locations/global/recognizers/_"
        
        self.streaming_config = cloud_speech.StreamingRecognitionConfig(
            config=cloud_speech.RecognitionConfig(
                language_codes=LANGUAGES,
                model="latest_short",
                explicit_decoding_config=cloud_speech.ExplicitDecodingConfig(
                    encoding=cloud_speech.ExplicitDecodingConfig.AudioEncoding.LINEAR16,
                    sample_rate_hertz=RATE,
                    audio_channel_count=1
                ),
                features=cloud_speech.RecognitionFeatures(),
            ),
        )
        
        self.conversation_buffer = []
        self.last_extraction_time = time.time()
        self.is_listening = False
        self.stream = None
        self.audio = None
        self.callback = None
        
    def extract_medical_entities(self, transcript: str) -> dict:
        response = self.openai_client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": "You extract structured medical information from clinical conversations."
                },
                {
                    "role": "user",
                    "content": f"""
Extract the following fields from the transcript:

- symptoms: array of symptom strings
- diseases: array of disease/diagnosis strings
- medicine_names: array of medicine names only (no dosage)
- medicine_timing: object mapping medicine name -> timing words (e.g., "morning and night", "afternoon and night")
- medicine_meal: object mapping medicine name -> "before food" or "after food"
- lab_reports: array of lab test names
- lab_city: string for lab location
- height: string (e.g., "5'8\"" or "172cm")
- weight: string with unit (e.g., "70kg")
- bp: string (e.g., "120/80")
- pulse: number
- spo2: number
- temperature: number
- referred_doctor: string
- referred_clinic: string
- total_fees: number
- paid_fees: number
- follow_up_date: string (ISO date, e.g., "YYYY-MM-DD")
- remarks: string
- take_image: boolean (true if user says take photo/image/picture or open camera)

Rules:
- Return valid JSON only
- Use null for missing information
- Do NOT guess or hallucinate
- Extract only what is explicitly mentioned

Transcript:
\"\"\"{transcript}\"\"\"
"""
                }
            ]
        )
        return json.loads(response.choices[0].message.content)
    
    def audio_request_generator(self):
        self.audio = pyaudio.PyAudio()
        self.stream = self.audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK,
        )
        print('🎤 Audio stream opened')
        
        yield cloud_speech.StreamingRecognizeRequest(
            recognizer=self.recognizer,
            streaming_config=self.streaming_config
        )
        
        while self.is_listening:
            try:
                data = self.stream.read(CHUNK, exception_on_overflow=False)
                yield cloud_speech.StreamingRecognizeRequest(audio=data)
            except Exception as e:
                print(f'❌ Audio read error: {e}')
                break
        print('🧹 Exiting audio_request_generator')
    
    def start_listening(self, callback):
        if self.is_listening:
            return
        
        self.callback = callback
        self.is_listening = True
        self.conversation_buffer = []
        self.last_extraction_time = time.time()
        
        def listen_thread():
            print('🎧 Listen thread started')
            while self.is_listening:
                try:
                    print('📡 Creating streaming recognize request...')
                    responses = self.speech_client.streaming_recognize(
                        requests=self.audio_request_generator(),
                    )
                    
                    print('🔊 Starting to process responses...')
                    for response in responses:
                        print(f'📨 Received response: {response}')
                        if not self.is_listening:
                            print('🛑 Stopping - is_listening is False')
                            break
                        
                        for result in response.results:
                            print(f'📋 Result: is_final={result.is_final}')
                            if not result.is_final:
                                continue
                            
                            alt = result.alternatives[0]
                            transcript = alt.transcript.strip()
                            
                            print(f'✍️ Transcript: "{transcript}"')
                            
                            if not transcript:
                                continue
                            
                            lang_code = getattr(result, 'language_code', 'unknown')
                            print(f"[{lang_code}] {transcript}")
                            self.conversation_buffer.append(transcript)
                            
                            # Send interim transcript
                            if self.callback:
                                print('📤 Sending transcript to callback')
                                self.callback({
                                    "type": "transcript",
                                    "text": transcript,
                                    "language": lang_code
                                })
                            
                            # Extract entities on first transcript and then periodically
                            now = time.time()
                            if (
                                now - self.last_extraction_time >= EXTRACTION_INTERVAL or
                                (self.conversation_buffer and now - self.last_extraction_time >= MIN_EXTRACTION_INTERVAL)
                            ):
                                print('🔍 Extracting entities...')
                                full_text = " ".join(self.conversation_buffer)
                                entities = self.extract_medical_entities(full_text)
                                
                                if self.callback:
                                    print('📤 Sending entities to callback')
                                    self.callback({
                                        "type": "entities",
                                        "data": entities
                                    })
                                
                                self.last_extraction_time = now
                    
                    print('🧹 Response stream ended')
                    if self.is_listening:
                        time.sleep(0.2)
                except Exception as e:
                    error_text = str(e)
                    if "Stream timed out" in error_text and self.is_listening:
                        print("⚠️ Stream timed out; restarting stream...")
                        time.sleep(0.3)
                        continue
                    print(f"❌ Error in voice listening: {e}")
                    import traceback
                    traceback.print_exc()
                    if self.callback:
                        self.callback({"type": "error", "message": str(e)})
                    break

            print('🧹 Cleaning up listen thread')
            self.cleanup()
        
        print('🚀 Starting listen thread...')
        threading.Thread(target=listen_thread, daemon=True).start()
        print('✅ Listen thread started')
    
    def stop_listening(self):
        self.is_listening = False
        self.cleanup()
    
    def cleanup(self):
        if self.stream:
            try:
                self.stream.stop_stream()
                self.stream.close()
            except:
                pass
            self.stream = None
        
        if self.audio:
            try:
                self.audio.terminate()
            except:
                pass
            self.audio = None
