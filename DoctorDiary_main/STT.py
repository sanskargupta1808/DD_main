#!/usr/bin/env python3
"""
Enhanced Medical Consultation WebSocket with Data Container Boxes
🏥 Frontend with autofilled data containers for Patient, Symptoms, Disease, Prescription
"""

import asyncio
import json
import speech_recognition as sr
import pyttsx3
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import uvicorn
import threading
import queue
import time
from typing import List
import os
import requests

class EnhancedMedicalWebSocket:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.microphone = None
        self.tts_engine = pyttsx3.init()
        self.connected_clients: List[WebSocket] = []
        self.message_queue = asyncio.Queue()
        self.sync_queue = queue.Queue()
        self.is_listening = False
        self.server_url = os.getenv("DD_SERVER_URL", "http://localhost:8080")
        
        # Medical consultation workflow
        self.consultation_stage = "patient"  # patient -> symptoms -> disease -> prescription
        self.current_consultation = {}
        
        # Configure TTS
        try:
            self.tts_engine.setProperty('rate', 150)
            self.tts_engine.setProperty('volume', 0.8)
        except:
            pass
        
        # Configure recognizer
        self.recognizer.energy_threshold = 200
        self.recognizer.dynamic_energy_threshold = True
        self.recognizer.pause_threshold = 0.8

    def post_transcript_to_server(self, text: str, language: str) -> None:
        """Send transcript to Doctor Diary server for NLP extraction"""
        payload = {
            "text": text,
            "language": language,
            "stage": self.consultation_stage
        }

        try:
            requests.post(
                f"{self.server_url}/api/nlp/extract",
                json=payload,
                timeout=10
            )
        except Exception as exc:
            print(f"❌ Failed to post transcript to server: {exc}")
    
    def extract_medical_data(self, text, language="unknown"):
        """Extract clean medical data from speech text with multi-language support"""
        if not text.strip():
            return None
        
        print(f"🔍 DEBUG: Original text: '{text}', Language: {language}")
        
        # Multi-language filler words removal
        filler_words = {
            'hindi': [
                'मेरा नाम', 'मेरा नाम है', 'मुझे', 'मुझे लगता है', 'यह', 'यह है', 'दें', 'दे दो',
                'मरीज़ का नाम', 'रोगी का नाम', 'लक्षण हैं', 'बीमारी है', 'दवा दें'
            ],
            'english': [
                'my name is', 'patient name is', 'name is', 'i have', 'patient has', 
                'symptoms are', 'this is', 'disease is', 'diagnosis is', 'give', 'take', 
                'prescription is', 'treatment is', 'the patient', 'patient is'
            ],
            'gujarati': [
                'મારું નામ', 'મારું નામ છે', 'મને', 'મને લાગે છે', 'આ', 'આ છે', 'આપો', 'આપી દો',
                'દર્દીનું નામ', 'લક્ષણો છે', 'બીમારી છે', 'દવા આપો'
            ],
            'tamil': [
                'என் பெயர்', 'என் பெயர் தான்', 'எனக்கு', 'எனக்கு தோன்றுகிறது', 'இது', 'இது தான்', 
                'கொடுங்கள்', 'நோயாளியின் பெயர்', 'அறிகுறிகள்', 'நோய்', 'மருந்து கொடுங்கள்'
            ],
            'telugu': [
                'నా పేరు', 'నా పేరు అది', 'నాకు', 'నాకు అనిపిస్తుంది', 'ఇది', 'ఇది అది', 
                'ఇవ్వండి', 'రోగి పేరు', 'లక్షణాలు', 'వ్యాధి', 'మందు ఇవ్వండి'
            ]
        }
        
        # Get language-specific filler words
        lang_fillers = filler_words.get(language, filler_words['english'])
        
        # Remove filler words
        cleaned_text = text.lower()
        for filler in lang_fillers:
            cleaned_text = cleaned_text.replace(filler.lower(), '').strip()
        
        print(f"🔍 DEBUG: After cleaning: '{cleaned_text}'")
        
        # If nothing left after cleaning, return original text (but cleaned)
        if not cleaned_text or len(cleaned_text) < 2:
            # Return original text without common filler words
            result = text.strip()
            print(f"🔍 DEBUG: Using original text: '{result}'")
            return result
        
        print(f"🔍 DEBUG: Final result: '{cleaned_text}'")
        return cleaned_text
    
    def reset_consultation(self):
        """Reset consultation for new session"""
        self.consultation_stage = "patient"
        self.current_consultation = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "patient": None,
            "symptoms": [],
            "disease": [],
            "prescription": []
        }
    
    def get_stage_prompt(self):
        """Get current stage prompt with multi-language examples"""
        prompts = {
            "patient": {
                "title": "👤 PATIENT NAME",
                "instruction": "Say patient name in any language",
                "examples": "Hindi: खुशबू | English: Khushbu | Gujarati: ખુશબુ | Tamil: குஷ்பு | Telugu: ఖుష్బు"
            },
            "symptoms": {
                "title": "🤒 SYMPTOMS", 
                "instruction": "Say symptoms in any language",
                "examples": "Hindi: बुखार सिरदर्द | English: fever headache | Gujarati: તાવ માથાનો દુખાવો | Tamil: காய்ச்சல் தலைவலி | Telugu: జ్వరం తలనొప్పి"
            },
            "disease": {
                "title": "🦠 DISEASE",
                "instruction": "Say disease diagnosis in any language",
                "examples": "Hindi: मलेरिया | English: malaria | Gujarati: મેલેરિયા | Tamil: மலேரியா | Telugu: మలేరియా"
            },
            "prescription": {
                "title": "💊 PRESCRIPTION",
                "instruction": "Say prescription in any language",
                "examples": "Hindi: पैरासिटामोल आराम | English: paracetamol rest | Gujarati: પેરાસિટામોલ આરામ | Tamil: பாராசிட்டமால் ஓய்வு | Telugu: పారాసిటమాల్ విశ్రాంతి"
            }
        }
        return prompts.get(self.consultation_stage, prompts["patient"])
    
    def speak_async(self, text):
        """Asynchronous text-to-speech"""
        def speak_thread():
            try:
                self.tts_engine.say(text)
                self.tts_engine.runAndWait()
            except Exception as e:
                print(f"❌ TTS Error: {e}")
        
        threading.Thread(target=speak_thread, daemon=True).start()
    
    def setup_microphone(self):
        """Setup microphone"""
        try:
            if self.microphone is not None:
                return True
                
            self.microphone = sr.Microphone()
            
            if not hasattr(self, '_microphone_calibrated'):
                with self.microphone as source:
                    print("🔧 Calibrating microphone...")
                    self.recognizer.adjust_for_ambient_noise(source, duration=2)
                self._microphone_calibrated = True
                
            print(f"✅ Microphone ready! Energy threshold: {self.recognizer.energy_threshold}")
            return True
        except Exception as e:
            print(f"❌ Microphone setup failed: {e}")
            return False
    
    async def add_client(self, websocket: WebSocket):
        """Add new WebSocket client"""
        self.connected_clients.append(websocket)
        
        if not self.microphone:
            self.setup_microphone()
        
        # Initialize medical consultation
        self.reset_consultation()
        
        await self.send_to_client(websocket, {
            "type": "status",
            "message": "🏥 Enhanced Medical Consultation Connected",
            "timestamp": time.time()
        })
        
        # Send initial stage prompt
        await self.send_stage_prompt()
        
        print(f"✅ Client connected. Total clients: {len(self.connected_clients)}")
    
    def remove_client(self, websocket: WebSocket):
        """Remove WebSocket client"""
        if websocket in self.connected_clients:
            self.connected_clients.remove(websocket)
        print(f"❌ Client disconnected. Total clients: {len(self.connected_clients)}")
    
    async def send_to_client(self, websocket: WebSocket, message: dict):
        """Send message to specific client"""
        try:
            await websocket.send_text(json.dumps(message))
        except:
            self.remove_client(websocket)
    
    async def broadcast_message(self, message: dict):
        """Broadcast message to all connected clients"""
        if not self.connected_clients:
            return
            
        disconnected = []
        for client in self.connected_clients:
            try:
                await client.send_text(json.dumps(message))
            except:
                disconnected.append(client)
        
        for client in disconnected:
            self.remove_client(client)
    
    def queue_message_sync(self, message: dict):
        """Queue message from sync context"""
        try:
            self.sync_queue.put(message)
        except Exception as e:
            print(f"Error queuing message: {e}")
    
    async def process_sync_queue(self):
        """Process messages from sync queue"""
        while True:
            try:
                if not self.sync_queue.empty():
                    message = self.sync_queue.get_nowait()
                    await self.message_queue.put(message)
                await asyncio.sleep(0.1)
            except Exception as e:
                print(f"Error processing sync queue: {e}")
                await asyncio.sleep(1)
    
    async def process_messages(self):
        """Process queued messages"""
        while True:
            try:
                message = await self.message_queue.get()
                
                if message.get("type") == "advance_stage":
                    await asyncio.sleep(1)
                    await self.handle_stage_advance()
                else:
                    await self.broadcast_message(message)
                    
            except Exception as e:
                print(f"Error processing message: {e}")
                await asyncio.sleep(1)
    
    async def send_stage_prompt(self):
        """Send current stage prompt to clients"""
        prompt = self.get_stage_prompt()
        await self.broadcast_message({
            "type": "stage_prompt",
            "stage": self.consultation_stage,
            "title": prompt["title"],
            "instruction": prompt["instruction"],
            "examples": prompt["examples"],
            "timestamp": time.time()
        })
        
        print(f"\n🎯 {prompt['title']}")
        print(f"🎙️ {prompt['instruction']}")
        print(f"💡 Examples: {prompt['examples']}")
        print("🔴 LISTENING...")
        
        # TTS announcement
        self.speak_async(prompt["instruction"])
    
    def advance_consultation_stage(self):
        """Advance to next consultation stage"""
        stages = ["patient", "symptoms", "disease", "prescription"]
        current_index = stages.index(self.consultation_stage)
        
        if current_index < len(stages) - 1:
            self.consultation_stage = stages[current_index + 1]
            return True
        else:
            return False
    
    def save_consultation(self):
        """Save consultation to file"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"medical_consultation_{timestamp}.json"
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(self.current_consultation, f, ensure_ascii=False, indent=2)
            return filename
        except Exception as e:
            print(f"❌ Error saving consultation: {e}")
            return None
    
    async def handle_stage_advance(self):
        """Handle advancing to next stage"""
        await self.send_stage_prompt()
    
    def start_live_recognition(self):
        """Start live speech recognition with multi-language support"""
        if not self.microphone:
            print("❌ Microphone not setup!")
            return False
        
        if hasattr(self, 'stop_listening') and self.is_listening:
            self.stop_listening_service()
            time.sleep(0.5)
        
        def audio_callback(recognizer, audio):
            """Process audio for medical consultation with multi-language support"""
            try:
                self.queue_message_sync({
                    "type": "processing",
                    "message": "🔄 Processing speech (Hindi, English, Gujarati, Tamil, Telugu)...",
                    "timestamp": time.time()
                })
                
                # Multi-language recognition with priority order
                recognized_text = None
                language = "unknown"
                language_codes = [
                    ('hi-IN', 'Hindi', '🇮🇳'),
                    ('en-IN', 'English', '🇺🇸'), 
                    ('gu-IN', 'Gujarati', '🇮🇳'),
                    ('ta-IN', 'Tamil', '🇮🇳'),
                    ('te-IN', 'Telugu', '🇮🇳')
                ]
                
                # Try each language until we get a result
                for lang_code, lang_name, flag in language_codes:
                    if recognized_text:
                        break
                        
                    try:
                        text = recognizer.recognize_google(audio, language=lang_code)
                        if text and text.strip():
                            recognized_text = text
                            language = lang_name.lower()
                            print(f"{flag} {lang_name}: {text}")
                            
                            # Send language detection update
                            self.queue_message_sync({
                                "type": "language_detected",
                                "language": lang_name,
                                "flag": flag,
                                "text": text,
                                "timestamp": time.time()
                            })
                            break
                    except Exception as lang_error:
                        print(f"❌ {lang_name} recognition failed: {lang_error}")
                        continue
                
                if not recognized_text:
                    self.queue_message_sync({
                        "type": "no_speech",
                        "message": "💭 No clear speech detected in any supported language",
                        "timestamp": time.time()
                    })
                    return
                
                # Process medical consultation stage
                self.process_medical_consultation(recognized_text, language)
                        
            except Exception as e:
                self.queue_message_sync({
                    "type": "error",
                    "message": f"❌ Error: {str(e)}",
                    "timestamp": time.time()
                })
                print(f"❌ Audio callback error: {e}")
        
        try:
            background_recognizer = sr.Recognizer()
            background_recognizer.energy_threshold = self.recognizer.energy_threshold
            background_recognizer.dynamic_energy_threshold = self.recognizer.dynamic_energy_threshold
            background_recognizer.pause_threshold = self.recognizer.pause_threshold
            
            self.stop_listening = background_recognizer.listen_in_background(
                self.microphone, 
                audio_callback,
                phrase_time_limit=5
            )
            self.is_listening = True
            
            self.queue_message_sync({
                "type": "live_status",
                "message": "🔴 LIVE - Enhanced medical consultation active",
                "timestamp": time.time()
            })
            
            print("🔴 LIVE enhanced medical consultation started")
            return True
        except Exception as e:
            print(f"❌ Failed to start listening: {e}")
            return False
    
    def process_medical_consultation(self, text, language):
        """Process speech for medical consultation workflow with multi-language support"""
        print(f"\n🎤 RAW INPUT: '{text}' (Language: {language})")
        print(f"📊 Current Stage: {self.consultation_stage}")

        threading.Thread(
            target=self.post_transcript_to_server,
            args=(text, language),
            daemon=True
        ).start()
        
        extracted_data = self.extract_medical_data(text, language)
        stage_completed = False
        
        print(f"📝 EXTRACTED DATA: '{extracted_data}'")
        
        # Language flags for display
        language_flags = {
            'hindi': '🇮🇳 हिंदी',
            'english': '🇺🇸 English', 
            'gujarati': '🇮🇳 ગુજરાતી',
            'tamil': '🇮🇳 தமிழ்',
            'telugu': '🇮🇳 తెలుగు'
        }
        
        lang_display = language_flags.get(language, f'🌐 {language.title()}')
        
        print(f"� Processing {self.consultation_stage.upper()}: '{text}' ({lang_display})")
        
        if self.consultation_stage == "patient":
            if extracted_data and len(extracted_data.strip()) > 0:
                self.current_consultation["patient"] = extracted_data
                self.queue_message_sync({
                    "type": "patient_identified",
                    "extracted_data": extracted_data,
                    "original_text": text,
                    "language": language,
                    "language_display": lang_display,
                    "stage": self.consultation_stage,
                    "timestamp": time.time()
                })
                print(f"✅ PATIENT COMPLETED: {extracted_data} ({lang_display})")
                self.speak_async(f"Patient identified: {extracted_data}")
                stage_completed = True
            else:
                print(f"❌ PATIENT FAILED: No valid data extracted from '{text}'")
        
        elif self.consultation_stage == "symptoms":
            if extracted_data and len(extracted_data.strip()) > 0:
                self.current_consultation["symptoms"].append(extracted_data)
                self.queue_message_sync({
                    "type": "symptoms_detected",
                    "extracted_data": extracted_data,
                    "original_text": text,
                    "language": language,
                    "language_display": lang_display,
                    "stage": self.consultation_stage,
                    "timestamp": time.time()
                })
                print(f"✅ SYMPTOMS COMPLETED: {extracted_data} ({lang_display})")
                self.speak_async(f"Symptoms recorded: {extracted_data}")
                stage_completed = True
            else:
                print(f"❌ SYMPTOMS FAILED: No valid data extracted from '{text}'")
        
        elif self.consultation_stage == "disease":
            if extracted_data and len(extracted_data.strip()) > 0:
                self.current_consultation["disease"].append(extracted_data)
                self.queue_message_sync({
                    "type": "disease_diagnosed",
                    "extracted_data": extracted_data,
                    "original_text": text,
                    "language": language,
                    "language_display": lang_display,
                    "stage": self.consultation_stage,
                    "timestamp": time.time()
                })
                print(f"✅ DISEASE COMPLETED: {extracted_data} ({lang_display})")
                self.speak_async(f"Disease diagnosed: {extracted_data}")
                stage_completed = True
            else:
                print(f"❌ DISEASE FAILED: No valid data extracted from '{text}'")
        
        elif self.consultation_stage == "prescription":
            if extracted_data and len(extracted_data.strip()) > 0:
                self.current_consultation["prescription"].append(extracted_data)
                self.queue_message_sync({
                    "type": "prescription_recorded",
                    "extracted_data": extracted_data,
                    "original_text": text,
                    "language": language,
                    "language_display": lang_display,
                    "stage": self.consultation_stage,
                    "timestamp": time.time()
                })
                print(f"✅ PRESCRIPTION COMPLETED: {extracted_data} ({lang_display})")
                self.speak_async(f"Prescription recorded: {extracted_data}")
                stage_completed = True
            else:
                print(f"❌ PRESCRIPTION FAILED: No valid data extracted from '{text}'")
        
        # Auto-advance to next stage
        if stage_completed:
            print(f"🚀 ADVANCING FROM {self.consultation_stage.upper()}")
            if self.advance_consultation_stage():
                print(f"➡️ ADVANCED TO {self.consultation_stage.upper()}")
                self.queue_message_sync({
                    "type": "advance_stage",
                    "next_stage": self.consultation_stage,
                    "timestamp": time.time()
                })
            else:
                print("🎉 ALL STAGES COMPLETED - FINISHING CONSULTATION")
                self.complete_consultation()
        else:
            print(f"⚠️ STAGE NOT COMPLETED - STAYING IN {self.consultation_stage.upper()}")
    
    def complete_consultation(self):
        """Complete the medical consultation"""
        filename = self.save_consultation()
        
        self.queue_message_sync({
            "type": "consultation_complete",
            "consultation": self.current_consultation,
            "filename": filename,
            "message": "🎉 Medical consultation completed!",
            "timestamp": time.time()
        })
        
        print(f"\n🎉 CONSULTATION COMPLETED!")
        print(f"📊 Summary:")
        print(f"   👤 Patient: {self.current_consultation.get('patient', 'Unknown')}")
        print(f"   🤒 Symptoms: {', '.join(self.current_consultation.get('symptoms', []))}")
        print(f"   🦠 Disease: {', '.join(self.current_consultation.get('disease', []))}")
        print(f"   💊 Prescription: {', '.join(self.current_consultation.get('prescription', []))}")
        if filename:
            print(f"💾 Saved to: {filename}")
        
        self.speak_async("Medical consultation completed successfully")
        
        # Reset for next consultation
        self.reset_consultation()
    
    def stop_listening_service(self):
        """Stop speech recognition"""
        try:
            if hasattr(self, 'stop_listening') and self.is_listening:
                self.stop_listening(wait_for_stop=False)
                self.is_listening = False
                print("🛑 Speech recognition stopped")
                delattr(self, 'stop_listening')
        except Exception as e:
            print(f"❌ Error stopping listening: {e}")
            self.is_listening = False

# Initialize handler
medical_handler = EnhancedMedicalWebSocket()

# FastAPI app
app = FastAPI(title="Enhanced Medical Consultation with Data Containers")

@app.get("/")
async def get_homepage():
    """Enhanced web interface with data container boxes"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>🏥 Enhanced Medical Consultation - Data Container Boxes</title>
        <meta charset="UTF-8">
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                max-width: 1400px;
                margin: 0 auto;
                padding: 20px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
            }
            .container {
                background: white;
                padding: 30px;
                border-radius: 15px;
                box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            }
            h1 {
                text-align: center;
                color: #333;
                margin-bottom: 30px;
            }
            .status {
                padding: 15px;
                border-radius: 8px;
                margin: 15px 0;
                font-weight: bold;
                text-align: center;
            }
            .connected { 
                background: linear-gradient(45deg, #4CAF50, #45a049);
                color: white;
            }
            .disconnected { 
                background: linear-gradient(45deg, #f44336, #da190b);
                color: white;
            }
            
            /* Data Container Boxes */
            .data-containers {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                gap: 20px;
                margin: 30px 0;
            }
            .data-container {
                background: #f8f9fa;
                border: 3px solid #e9ecef;
                border-radius: 15px;
                padding: 20px;
                transition: all 0.3s ease;
                min-height: 120px;
            }
            .data-container.active {
                border-color: #FF9800;
                background: linear-gradient(45deg, #fff3e0, #ffcc80);
                transform: scale(1.02);
                animation: pulse 2s infinite;
            }
            .data-container.filled {
                border-color: #4CAF50;
                background: linear-gradient(45deg, #e8f5e8, #c8e6c9);
            }
            .data-container-header {
                display: flex;
                align-items: center;
                margin-bottom: 15px;
            }
            .data-container-icon {
                font-size: 32px;
                margin-right: 15px;
            }
            .data-container-title {
                font-size: 20px;
                font-weight: bold;
                color: #333;
            }
            .data-container-content {
                background: white;
                border-radius: 10px;
                padding: 15px;
                min-height: 60px;
                border: 2px dashed #ddd;
                display: flex;
                align-items: center;
                justify-content: center;
                transition: all 0.3s ease;
            }
            .data-container.filled .data-container-content {
                border: 2px solid #4CAF50;
                background: #f1f8e9;
            }
            .data-container.active .data-container-content {
                border: 2px solid #FF9800;
                background: #fff8e1;
                animation: glow 1.5s infinite;
            }
            .data-placeholder {
                color: #999;
                font-style: italic;
                text-align: center;
            }
            .data-value {
                color: #2e7d32;
                font-weight: bold;
                font-size: 16px;
                text-align: center;
            }
            .data-details {
                margin-top: 10px;
                font-size: 12px;
                color: #666;
                text-align: center;
            }
            
            /* Workflow Progress Bar */
            .workflow-progress {
                background: #f8f9fa;
                padding: 20px;
                border-radius: 15px;
                margin: 25px 0;
                border: 2px solid #28a745;
            }
            .progress-title {
                text-align: center;
                font-size: 18px;
                font-weight: bold;
                color: #28a745;
                margin-bottom: 15px;
            }
            .progress-bar {
                display: flex;
                align-items: center;
                justify-content: space-between;
                margin: 15px 0;
            }
            .progress-step {
                flex: 1;
                text-align: center;
                padding: 10px;
                border-radius: 20px;
                margin: 0 5px;
                transition: all 0.3s ease;
                border: 2px solid #ddd;
                background: white;
            }
            .progress-step.active {
                border-color: #FF9800;
                background: #fff3e0;
                transform: scale(1.05);
            }
            .progress-step.completed {
                border-color: #4CAF50;
                background: #e8f5e8;
            }
            .progress-arrow {
                font-size: 20px;
                color: #666;
            }
            
            /* Multi-Language Support */
            .language-support {
                background: linear-gradient(45deg, #e8f5e8, #c8e6c9);
                padding: 20px;
                border-radius: 15px;
                margin: 25px 0;
                border: 2px solid #4CAF50;
                text-align: center;
            }
            .language-title {
                font-size: 18px;
                font-weight: bold;
                color: #2e7d32;
                margin-bottom: 15px;
            }
            .language-flags {
                display: flex;
                flex-wrap: wrap;
                justify-content: center;
                gap: 15px;
                margin: 15px 0;
            }
            .language-flag {
                background: white;
                padding: 8px 15px;
                border-radius: 20px;
                border: 2px solid #4CAF50;
                font-weight: bold;
                font-size: 14px;
                color: #2e7d32;
                transition: all 0.3s ease;
            }
            .language-flag:hover {
                transform: scale(1.05);
                box-shadow: 0 3px 10px rgba(76, 175, 80, 0.3);
            }
            .language-note {
                font-style: italic;
                color: #666;
                margin-top: 10px;
            }
            
            /* Current Stage Info */
            .current-stage {
                background: linear-gradient(45deg, #e3f2fd, #bbdefb);
                padding: 20px;
                border-radius: 12px;
                margin: 20px 0;
                border-left: 5px solid #2196F3;
                text-align: center;
            }
            .stage-instruction {
                font-size: 18px;
                font-weight: bold;
                color: #1976D2;
                margin-bottom: 10px;
            }
            .stage-examples {
                font-style: italic;
                color: #666;
            }
            
            /* Listening Indicator */
            .listening-indicator {
                display: none;
                text-align: center;
                padding: 15px;
                background: linear-gradient(45deg, #ffebee, #ffcdd2);
                border-radius: 10px;
                margin: 15px 0;
                animation: pulse 1.5s infinite;
                font-weight: bold;
                color: #d32f2f;
            }
            .listening-indicator.active {
                display: block;
            }
            
            .controls {
                text-align: center;
                margin: 25px 0;
            }
            button {
                background: linear-gradient(45deg, #2196F3, #1976D2);
                color: white;
                border: none;
                padding: 12px 25px;
                border-radius: 25px;
                cursor: pointer;
                margin: 5px;
                font-size: 16px;
                transition: all 0.3s;
            }
            button:hover { 
                transform: translateY(-2px);
                box-shadow: 0 5px 15px rgba(0,0,0,0.2);
            }
            button:disabled { 
                background: #ccc;
                cursor: not-allowed;
                transform: none;
            }
            .start-button { background: linear-gradient(45deg, #4CAF50, #45a049); }
            .stop-button { background: linear-gradient(45deg, #f44336, #d32f2f); }
            
            .results {
                max-height: 300px;
                overflow-y: auto;
                border: 2px solid #e0e0e0;
                padding: 15px;
                border-radius: 10px;
                background: #fafafa;
                margin-top: 20px;
            }
            .message {
                background: white;
                border-left: 5px solid #2196F3;
                padding: 10px;
                margin: 8px 0;
                border-radius: 5px;
                font-size: 14px;
            }
            
            @keyframes pulse {
                0% { transform: scale(1); }
                50% { transform: scale(1.02); }
                100% { transform: scale(1); }
            }
            @keyframes glow {
                0% { box-shadow: 0 0 5px rgba(255, 152, 0, 0.5); }
                50% { box-shadow: 0 0 20px rgba(255, 152, 0, 0.8); }
                100% { box-shadow: 0 0 5px rgba(255, 152, 0, 0.5); }
            }
            
            @media (max-width: 768px) {
                .data-containers {
                    grid-template-columns: 1fr;
                }
                .progress-bar {
                    flex-direction: column;
                }
                .progress-arrow {
                    transform: rotate(90deg);
                    margin: 10px 0;
                }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🏥 Enhanced Medical Consultation - Data Container Boxes</h1>
            
            <div id="status" class="status disconnected">
                ❌ Disconnected
            </div>
            
            <!-- Workflow Progress Bar -->
            <div class="workflow-progress">
                <div class="progress-title">📋 Medical Consultation Progress</div>
                <div class="progress-bar">
                    <div class="progress-step active" id="progress-patient">👤 Patient</div>
                    <div class="progress-arrow">→</div>
                    <div class="progress-step" id="progress-symptoms">🤒 Symptoms</div>
                    <div class="progress-arrow">→</div>
                    <div class="progress-step" id="progress-disease">🦠 Disease</div>
                    <div class="progress-arrow">→</div>
                    <div class="progress-step" id="progress-prescription">💊 Prescription</div>
                </div>
            </div>
            
            <!-- Data Container Boxes -->
            <div class="data-containers">
                <div class="data-container active" id="container-patient">
                    <div class="data-container-header">
                        <div class="data-container-icon">👤</div>
                        <div class="data-container-title">Patient Name</div>
                    </div>
                    <div class="data-container-content" id="content-patient">
                        <div class="data-placeholder">Waiting for patient name...</div>
                    </div>
                    <div class="data-details" id="details-patient">Speak patient name to autofill</div>
                </div>
                
                <div class="data-container" id="container-symptoms">
                    <div class="data-container-header">
                        <div class="data-container-icon">🤒</div>
                        <div class="data-container-title">Symptoms</div>
                    </div>
                    <div class="data-container-content" id="content-symptoms">
                        <div class="data-placeholder">Waiting for symptoms...</div>
                    </div>
                    <div class="data-details" id="details-symptoms">Describe patient symptoms</div>
                </div>
                
                <div class="data-container" id="container-disease">
                    <div class="data-container-header">
                        <div class="data-container-icon">🦠</div>
                        <div class="data-container-title">Disease</div>
                    </div>
                    <div class="data-container-content" id="content-disease">
                        <div class="data-placeholder">Waiting for diagnosis...</div>
                    </div>
                    <div class="data-details" id="details-disease">State the disease diagnosis</div>
                </div>
                
                <div class="data-container" id="container-prescription">
                    <div class="data-container-header">
                        <div class="data-container-icon">💊</div>
                        <div class="data-container-title">Prescription</div>
                    </div>
                    <div class="data-container-content" id="content-prescription">
                        <div class="data-placeholder">Waiting for treatment...</div>
                    </div>
                    <div class="data-details" id="details-prescription">Provide treatment plan</div>
                </div>
            </div>
            
            <!-- Multi-Language Support Info -->
            <div class="language-support">
                <div class="language-title">🌐 Multi-Language Support</div>
                <div class="language-flags">
                    <span class="language-flag">🇮🇳 हिंदी Hindi</span>
                    <span class="language-flag">🇺🇸 English</span>
                    <span class="language-flag">🇮🇳 ગુજરાતી Gujarati</span>
                    <span class="language-flag">🇮🇳 தமிழ் Tamil</span>
                    <span class="language-flag">🇮🇳 తెలుగు Telugu</span>
                </div>
                <div class="language-note">Speak in any of these languages - automatic detection!</div>
            </div>
            
            <!-- Current Stage Info -->
            <div class="current-stage" id="currentStage">
                <div class="stage-instruction" id="stageInstruction">Ready to start consultation...</div>
                <div class="stage-examples" id="stageExamples">Connect and click "Start Recognition" to begin</div>
            </div>
            
            <!-- Listening Indicator -->
            <div class="listening-indicator" id="listeningIndicator">
                🎤 LISTENING... Speak now!
            </div>
            
            <div class="controls">
                <button id="connectBtn" onclick="connect()">🔗 Connect</button>
                <button id="disconnectBtn" onclick="disconnect()" disabled>🔌 Disconnect</button>
                <button onclick="startRecognition()" class="start-button">🎙️ Start Recognition</button>
                <button onclick="stopRecognition()" class="stop-button">⏹️ Stop</button>
                <button onclick="clearAll()" style="background: linear-gradient(45deg, #9C27B0, #7B1FA2);">🗑️ Clear All</button>
            </div>
            
            <div>
                <h3>📝 Live Consultation Log:</h3>
                <div id="results" class="results">
                    <p style="text-align: center; color: #666;">
                        Connect to start medical consultation...
                    </p>
                </div>
            </div>
        </div>

        <script>
            let ws = null;
            let currentStage = 'patient';
            let consultationData = {
                patient: null,
                symptoms: [],
                disease: [],
                prescription: []
            };

            function updateStatus(message, className) {
                const status = document.getElementById('status');
                status.textContent = message;
                status.className = 'status ' + className;
            }
            
            function updateDataContainer(stage, data, language = '', originalText = '') {
                const container = document.getElementById(`container-${stage}`);
                const content = document.getElementById(`content-${stage}`);
                const details = document.getElementById(`details-${stage}`);
                
                // Update container state
                container.className = 'data-container filled';
                
                // Update content
                content.innerHTML = `<div class="data-value">${data}</div>`;
                
                // Update details
                if (language && originalText) {
                    details.innerHTML = `✅ Recorded (${language}): "${originalText}"`;
                } else {
                    details.innerHTML = `✅ Data recorded successfully`;
                }
            }
            
            function setActiveContainer(stage) {
                // Reset all containers
                const containers = ['patient', 'symptoms', 'disease', 'prescription'];
                containers.forEach(s => {
                    const container = document.getElementById(`container-${s}`);
                    if (consultationData[s] && (Array.isArray(consultationData[s]) ? consultationData[s].length > 0 : consultationData[s])) {
                        container.className = 'data-container filled';
                    } else if (s === stage) {
                        container.className = 'data-container active';
                    } else {
                        container.className = 'data-container';
                    }
                });
            }
            
            function updateProgressBar(stage) {
                const stages = ['patient', 'symptoms', 'disease', 'prescription'];
                const currentIndex = stages.indexOf(stage);
                
                stages.forEach((s, index) => {
                    const progressStep = document.getElementById(`progress-${s}`);
                    if (index < currentIndex) {
                        progressStep.className = 'progress-step completed';
                    } else if (index === currentIndex) {
                        progressStep.className = 'progress-step active';
                    } else {
                        progressStep.className = 'progress-step';
                    }
                });
            }
            
            function updateCurrentStage(title, instruction, examples) {
                document.getElementById('stageInstruction').textContent = instruction;
                document.getElementById('stageExamples').textContent = `Examples: ${examples}`;
            }
            
            function showListening(show) {
                const indicator = document.getElementById('listeningIndicator');
                if (show) {
                    indicator.classList.add('active');
                } else {
                    indicator.classList.remove('active');
                }
            }

            function addResult(text, type) {
                const results = document.getElementById('results');
                const div = document.createElement('div');
                div.className = `message`;
                
                const timestamp = new Date().toLocaleTimeString();
                div.innerHTML = `<strong>[${timestamp}]:</strong> ${text}`;
                
                results.appendChild(div);
                results.scrollTop = results.scrollHeight;
                
                // Keep only last 20 messages
                const messages = results.querySelectorAll('.message');
                if (messages.length > 20) {
                    messages[0].remove();
                }
            }

            function connect() {
                if (ws) return;
                
                updateStatus('🔄 Connecting...', 'disconnected');
                
                ws = new WebSocket('ws://localhost:8011/ws');
                
                ws.onopen = function() {
                    updateStatus('✅ Connected - Enhanced consultation ready', 'connected');
                    document.getElementById('connectBtn').disabled = true;
                    document.getElementById('disconnectBtn').disabled = false;
                };
                
                ws.onmessage = function(event) {
                    const data = JSON.parse(event.data);
                    
                    if (data.type === 'stage_prompt') {
                        currentStage = data.stage;
                        updateCurrentStage(data.title, data.instruction, data.examples);
                        setActiveContainer(data.stage);
                        updateProgressBar(data.stage);
                        addResult(`🎯 ${data.title}: ${data.instruction}`, 'stage_prompt');
                    } else if (data.type === 'language_detected') {
                        addResult(`${data.flag} Language detected: ${data.language} - "${data.text}"`, 'language_detected');
                    } else if (data.type === 'patient_identified') {
                        consultationData.patient = data.extracted_data;
                        const langDisplay = data.language_display || data.language;
                        updateDataContainer('patient', data.extracted_data, langDisplay, data.original_text);
                        addResult(`👤 Patient: ${data.extracted_data} (${langDisplay})`, 'patient_identified');
                        showListening(false);
                    } else if (data.type === 'symptoms_detected') {
                        consultationData.symptoms.push(data.extracted_data);
                        const symptomsText = consultationData.symptoms.join(', ');
                        const langDisplay = data.language_display || data.language;
                        updateDataContainer('symptoms', symptomsText, langDisplay, data.original_text);
                        addResult(`� Symptoms: ${data.extracted_data} (${langDisplay})`, 'symptoms_detected');
                        showListening(false);
                    } else if (data.type === 'disease_diagnosed') {
                        consultationData.disease.push(data.extracted_data);
                        const diseaseText = consultationData.disease.join(', ');
                        const langDisplay = data.language_display || data.language;
                        updateDataContainer('disease', diseaseText, langDisplay, data.original_text);
                        addResult(`🦠 Disease: ${data.extracted_data} (${langDisplay})`, 'disease_diagnosed');
                        showListening(false);
                    } else if (data.type === 'prescription_recorded') {
                        consultationData.prescription.push(data.extracted_data);
                        const prescriptionText = consultationData.prescription.join(', ');
                        const langDisplay = data.language_display || data.language;
                        updateDataContainer('prescription', prescriptionText, langDisplay, data.original_text);
                        addResult(`💊 Prescription: ${data.extracted_data} (${langDisplay})`, 'prescription_recorded');
                        showListening(false);
                    } else if (data.type === 'consultation_complete') {
                        const consultation = data.consultation;
                        addResult(`🎉 Consultation completed! Saved to: ${data.filename}`, 'consultation_complete');
                        
                        // Reset after 5 seconds
                        setTimeout(() => {
                            clearAll();
                        }, 5000);
                    } else if (data.type === 'processing') {
                        addResult(`� ${data.message}`, 'processing');
                        showListening(true);
                    } else if (data.type === 'no_speech') {
                        addResult(`❓ ${data.message}`, 'no_speech');
                        showListening(false);
                    } else if (data.type === 'live_status') {
                        addResult(`� ${data.message}`, 'live_status');
                        showListening(true);
                    } else if (data.type === 'error') {
                        addResult(`❌ ${data.message}`, 'error');
                        showListening(false);
                    } else if (data.type === 'status') {
                        addResult(`📋 ${data.message}`, 'status');
                    }
                };
                
                ws.onclose = function() {
                    updateStatus('❌ Disconnected', 'disconnected');
                    document.getElementById('connectBtn').disabled = false;
                    document.getElementById('disconnectBtn').disabled = true;
                    showListening(false);
                    ws = null;
                };
                
                ws.onerror = function(error) {
                    console.error('WebSocket error:', error);
                    addResult('WebSocket connection error', 'error');
                };
            }

            function startRecognition() {
                if (!ws) {
                    alert('Please connect first!');
                    return;
                }
                
                ws.send(JSON.stringify({
                    type: 'start_recognition'
                }));
            }

            function stopRecognition() {
                if (!ws) return;
                
                ws.send(JSON.stringify({
                    type: 'stop_recognition'
                }));
                showListening(false);
            }

            function disconnect() {
                if (ws) {
                    ws.close();
                }
            }

            function clearAll() {
                // Reset consultation data
                consultationData = { patient: null, symptoms: [], disease: [], prescription: [] };
                
                // Reset containers
                const containers = ['patient', 'symptoms', 'disease', 'prescription'];
                containers.forEach(stage => {
                    const container = document.getElementById(`container-${stage}`);
                    const content = document.getElementById(`content-${stage}`);
                    const details = document.getElementById(`details-${stage}`);
                    
                    if (stage === 'patient') {
                        container.className = 'data-container active';
                        content.innerHTML = '<div class="data-placeholder">Waiting for patient name...</div>';
                        details.innerHTML = 'Speak patient name to autofill';
                    } else {
                        container.className = 'data-container';
                        content.innerHTML = `<div class="data-placeholder">Waiting for ${stage}...</div>`;
                        details.innerHTML = `Describe patient ${stage}`;
                    }
                });
                
                // Reset progress bar
                updateProgressBar('patient');
                
                // Clear results
                document.getElementById('results').innerHTML = 
                    '<p style="text-align: center; color: #666;">Ready for next consultation...</p>';
                
                // Reset stage
                currentStage = 'patient';
                updateCurrentStage('Patient Name', 'Ready to start consultation', 'Connect and start recognition');
            }

            // Auto-connect on page load
            window.onload = function() {
                setTimeout(connect, 1000);
            };
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for enhanced medical consultation"""
    await websocket.accept()
    await medical_handler.add_client(websocket)
    
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message.get("type") == "start_recognition":
                success = medical_handler.start_live_recognition()
                await medical_handler.send_to_client(websocket, {
                    "type": "consultation_status",
                    "message": "🔴 Enhanced consultation started!" if success else "❌ Failed to start",
                    "timestamp": time.time()
                })
                
            elif message.get("type") == "stop_recognition":
                medical_handler.stop_listening_service()
                await medical_handler.send_to_client(websocket, {
                    "type": "consultation_status",
                    "message": "⏹️ Consultation stopped",
                    "timestamp": time.time()
                })
                
    except WebSocketDisconnect:
        medical_handler.remove_client(websocket)

async def startup():
    """Initialize service"""
    print("🚀 Starting Enhanced Medical Consultation with Data Containers...")
    asyncio.create_task(medical_handler.process_messages())
    asyncio.create_task(medical_handler.process_sync_queue())
    print("✅ Enhanced system ready!")

@app.on_event("startup")
async def startup_event():
    await startup()

if __name__ == "__main__":
    print("🏥 Enhanced Medical Consultation with Data Container Boxes")
    print("🌐 Open http://localhost:8011 in your browser")
    print("📋 Features:")
    print("   • Visual Data Container Boxes")
    print("   • Real-time Autofill")
    print("   • Interactive Progress Tracking")
    print("   • Multi-language Speech Recognition (Hindi, English, Gujarati, Tamil, Telugu)")
    uvicorn.run(app, host="0.0.0.0", port=8011)
