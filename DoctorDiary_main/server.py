
#!/usr/bin/env python3
from __future__ import annotations

from flask import Flask, request, jsonify, send_from_directory, send_file
from flask_cors import CORS
import base64
import json
import os
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple
import time
import re
import requests
import threading

try:
    import speech_recognition as sr
except Exception:
    sr = None

try:
    import pyttsx3
except Exception:
    pyttsx3 = None

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent / ".env")
except ImportError:
    pass

app = Flask(__name__)
CORS(app)

if not os.getenv("GROQ_API_KEY"):
    print(
        "⚠️  GROQ_API_KEY is not set. Copy .env.example to .env and add your key "
        "(medical NLP extraction will fail without it)."
    )

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / 'doctordiary.sqlite3'
UPLOADS_DIR = BASE_DIR / 'Uploads'

print(f'✅ Using main database at: {DB_PATH}')

_TABLE_COLUMNS: Dict[str, set] = {}
_LATEST_NLP: Dict[str, Any] = {"timestamp": 0.0, "data": None, "text": None, "stage": None, "language": None}
_STT_STATUS: Dict[str, Any] = {"active": False, "message": "idle"}


class EmbeddedSTT:
    def __init__(self):
        self.recognizer = sr.Recognizer() if sr else None
        self.microphone = None
        try:
            self.tts_engine = pyttsx3.init() if pyttsx3 else None
        except Exception:
            self.tts_engine = None
        self.is_listening = False
        self.stop_listening = None
        self.consultation_stage = "patient"

        if self.recognizer:
            self.recognizer.energy_threshold = 200
            self.recognizer.dynamic_energy_threshold = True
            self.recognizer.pause_threshold = 0.8

        if self.tts_engine:
            try:
                self.tts_engine.setProperty('rate', 150)
                self.tts_engine.setProperty('volume', 0.8)
            except Exception:
                pass

    def speak_async(self, text: str) -> None:
        if not self.tts_engine:
            return

        def _speak():
            try:
                self.tts_engine.say(text)
                self.tts_engine.runAndWait()
            except Exception:
                return

        threading.Thread(target=_speak, daemon=True).start()

    def setup_microphone(self) -> bool:
        if not sr or not self.recognizer:
            return False
        if self.microphone is not None:
            return True
        try:
            self.microphone = sr.Microphone()
            if not hasattr(self, '_microphone_calibrated'):
                with self.microphone as source:
                    print("🎙️ STT: Calibrating microphone...")
                    self.recognizer.adjust_for_ambient_noise(source, duration=2)
                self._microphone_calibrated = True
            print("✅ STT: Microphone ready")
            return True
        except Exception:
            return False

    def extract_medical_data(self, text: str, language: str) -> Optional[str]:
        if not text or not text.strip():
            return None

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

        lang_fillers = filler_words.get(language, filler_words['english'])
        cleaned = text.lower()
        for filler in lang_fillers:
            cleaned = cleaned.replace(filler.lower(), '').strip()

        if not cleaned or len(cleaned) < 2:
            return text.strip()
        return cleaned

    def process_transcript(self, text: str, language: str) -> None:
        print(f"🎤 STT: Transcript='{text}' (language={language}) stage={self.consultation_stage}")
        extracted = self.extract_medical_data(text, language)
        if extracted and extracted != text:
            print(f"🧹 STT: Cleaned='{extracted}'")
        payload = _groq_extract_medical(extracted or text, stage=self.consultation_stage, language=language)

        global _LATEST_NLP
        _LATEST_NLP = {
            'timestamp': time.time(),
            'data': payload,
            'text': text,
            'stage': self.consultation_stage,
            'language': language
        }
        stages = ["patient", "symptoms", "disease", "prescription"]
        if self.consultation_stage in stages:
            idx = stages.index(self.consultation_stage)
            if idx < len(stages) - 1:
                self.consultation_stage = stages[idx + 1]
            else:
                self.consultation_stage = "patient"
        print(f"➡️ STT: Next stage={self.consultation_stage}")

    def start_listening(self) -> bool:
        if not sr or not self.recognizer:
            _STT_STATUS.update({"active": False, "message": "speech_recognition not available"})
            return False
        if not self.setup_microphone():
            _STT_STATUS.update({"active": False, "message": "microphone not available"})
            return False
        if self.is_listening and self.stop_listening:
            print("ℹ️ STT: Already listening")
            return True

        def audio_callback(recognizer, audio):
            try:
                recognized_text = None
                language = "unknown"
                language_codes = [
                    ('hi-IN', 'hindi'),
                    ('en-IN', 'english'),
                    ('gu-IN', 'gujarati'),
                    ('ta-IN', 'tamil'),
                    ('te-IN', 'telugu')
                ]

                for lang_code, lang_name in language_codes:
                    if recognized_text:
                        break
                    try:
                        text = recognizer.recognize_google(audio, language=lang_code)
                        if text and text.strip():
                            recognized_text = text
                            language = lang_name
                            print(f"✅ STT: Recognized ({lang_name}) '{text}'")
                            break
                    except Exception:
                        continue

                if not recognized_text:
                    print("⚠️ STT: No speech detected")
                    return

                self.process_transcript(recognized_text, language)
            except Exception:
                return

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
            _STT_STATUS.update({"active": True, "message": "listening"})
            print("🔴 STT: Listening started")
            return True
        except Exception as exc:
            _STT_STATUS.update({"active": False, "message": f"failed: {exc}"})
            print(f"❌ STT: Failed to start: {exc}")
            return False

    def stop_listening_service(self) -> None:
        if self.stop_listening and self.is_listening:
            try:
                self.stop_listening(wait_for_stop=False)
            except Exception:
                pass
        self.stop_listening = None
        self.is_listening = False
        _STT_STATUS.update({"active": False, "message": "stopped"})
        print("🛑 STT: Listening stopped")


_STT_HANDLER = EmbeddedSTT()

# Voice Integration
try:
    from voice_integration import VoiceIntegration
    _VOICE_INTEGRATION = None
    _VOICE_CLIENTS = {}
    print('✅ VoiceIntegration module imported successfully')
except ImportError as e:
    print(f'❌ Failed to import VoiceIntegration: {e}')
    _VOICE_INTEGRATION = None
    _VOICE_CLIENTS = {}
except Exception as e:
    print(f'❌ Error importing VoiceIntegration: {e}')
    import traceback
    traceback.print_exc()
    _VOICE_INTEGRATION = None
    _VOICE_CLIENTS = {}
    _VOICE_CLIENTS = {}




def _extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    text = text.strip()
    if text.startswith('{') and text.endswith('}'):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def _groq_extract_medical(text: str, stage: Optional[str] = None, language: Optional[str] = None) -> Dict[str, Any]:
    api_key = os.getenv('GROQ_API_KEY')
    if not api_key:
        print('❌ NLP: GROQ_API_KEY is not set')
        return {"error": "GROQ_API_KEY is not set"}

    system_prompt = (
        "You are a medical data extraction AI. Extract structured medical information from transcripts. "
        "IMPORTANT: Always extract symptoms, diseases, prescriptions, and fees when mentioned. "
        "\n\n"
        "Return ONLY valid JSON with these keys:\n"
        "- patient_name: string or null\n"
        "- symptoms: array of symptom strings (e.g., ['fever', 'headache', 'cough'])\n"
        "- diseases: array of disease/diagnosis strings (e.g., ['flu', 'diabetes'])\n"
        "- prescriptions: array of objects with {name, type, dosage, timing, duration_days, quantity, instructions}\n"
        "- notes: string or null\n"
        "- follow_up_date: string or null\n"
        "- lab_reports: object with {lab_city, reports: [array of test names]}\n"
        "- height: string or null\n"
        "- weight: string or null\n"
        "- bp: string or null\n"
        "- pulse: string or null\n"
        "- spo2: string or null\n"
        "- temperature: string or null\n"
        "- referred_doctor: string or null (extract doctor name when patient is referred)\n"
        "- referred_hospital: string or null (extract clinic/hospital name when patient is referred)\n"
        "- fees: number or null (extract ANY mention of fees/charges/amount/payment, convert to number only)\n"
        "\n"
        "FEES EXTRACTION RULES:\n"
        "- Extract fees from: 'fees 500', 'charge 300', 'amount 400', 'consultation 500', '500 rupees', 'फीस 500'\n"
        "- Return ONLY the numeric value (e.g., 500, not '500 rupees')\n"
        "- If multiple numbers, extract the one related to fees/charges/payment\n"
        "- Common keywords: fees, charge, amount, payment, consultation, rupees, रुपये, फीस\n"
        "\n"
        "EXAMPLES:\n"
        "Input: 'Patient has fever and headache'\n"
        "Output: {\"symptoms\": [\"fever\", \"headache\"], \"diseases\": [], \"fees\": null, ...}\n"
        "\n"
        "Input: 'Fees 500 rupees'\n"
        "Output: {\"symptoms\": [], \"diseases\": [], \"fees\": 500, ...}\n"
        "\n"
        "Input: 'Charge 300'\n"
        "Output: {\"symptoms\": [], \"diseases\": [], \"fees\": 300, ...}\n"
        "\n"
        "Input: 'Patient has fever. Consultation fees 400.'\n"
        "Output: {\"symptoms\": [\"fever\"], \"diseases\": [], \"fees\": 400, ...}\n"
        "\n"
        "Input: 'फीस 500 रुपये'\n"
        "Output: {\"symptoms\": [], \"diseases\": [], \"fees\": 500, ...}\n"
        "\n"
        "Input: 'Refer to Dr. Smith at City Hospital'\n"
        "Output: {\"referred_doctor\": \"Dr. Smith\", \"referred_hospital\": \"City Hospital\", \"fees\": null, ...}\n"
        "\n"
        "CRITICAL: Extract ALL symptoms, diseases, and fees mentioned."
    )

    user_payload = {
        "transcript": text,
        "stage_hint": stage,
        "language_hint": language
    }

    payload = {
        "model": "llama-3.3-70b-versatile",  # Better model for medical extraction
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    "Extract medical data from this transcript. "
                    "Return ONLY valid minified JSON (no markdown, no commentary, no explanation). "
                    "IMPORTANT: Include 'fees' field (number or null) if any fees/charges/amount mentioned. "
                    f"Transcript: {text}\n"
                    f"Language: {language or 'unknown'}\n"
                    f"Stage: {stage or 'auto'}"
                )
            }
        ],
        "temperature": 0,
        "max_tokens": 500
    }

    try:
        print(f"🧠 NLP: Sending transcript to Groq (stage={stage}, language={language})")
        print(f"📝 NLP: Text='{text}'")
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            json=payload,
            timeout=20
        )
    except requests.RequestException as exc:
        return {"error": f"Groq request failed: {exc}"}

    if response.status_code != 200:
        print(f"❌ NLP: Groq error {response.status_code}: {response.text}")
        return {"error": f"Groq error {response.status_code}: {response.text}"}

    data = response.json()
    content = (
        data.get("choices", [{}])[0]
        .get("message", {})
        .get("content", "")
    )
    print(f"🤖 NLP: Groq response: {content}")
    parsed = _extract_json_from_text(content)
    if not parsed:
        print(f"❌ NLP: Failed to parse JSON from Groq: {content}")
        return {"error": "Failed to parse Groq JSON response", "raw": content}
    
    print(f"✅ NLP: Extraction success - {parsed}")
    return parsed


def get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    _ensure_tables(conn)
    return conn


def _get_table_columns(conn: sqlite3.Connection, table: str) -> set:
    if table in _TABLE_COLUMNS:
        return _TABLE_COLUMNS[table]
    rows = conn.execute(f'PRAGMA table_info({table})').fetchall()
    cols = {row['name'] for row in rows}
    _TABLE_COLUMNS[table] = cols
    return cols


def _insert_row(
    conn: sqlite3.Connection,
    table: str,
    data: Dict[str, Any],
    conflict: Optional[str] = None
) -> None:
    cols = _get_table_columns(conn, table)
    filtered = {k: v for k, v in data.items() if k in cols}
    if not filtered:
        return
    columns = ', '.join(filtered.keys())
    placeholders = ', '.join(['?'] * len(filtered))
    conflict_clause = f' OR {conflict}' if conflict else ''
    sql = f'INSERT{conflict_clause} INTO {table} ({columns}) VALUES ({placeholders})'
    conn.execute(sql, list(filtered.values()))


def _ensure_tables(conn: sqlite3.Connection) -> None:
    # Create TreatmentImages table if missing (required by Flutter bridge)
    conn.execute('''
        CREATE TABLE IF NOT EXISTS TreatmentImages(
            OfflineId TEXT PRIMARY KEY,
            TreatmentOfflineId TEXT,
            Id INTEGER,
            TreatmentId INTEGER,
            Image_Path TEXT,
            IsPrescriptionImage BOOLEAN,
            IsDeleted BOOLEAN,
            IsSync BOOLEAN,
            ThumbnailImage TEXT
        )
    ''')
    conn.commit()


def _convert_height_to_decimal(height_str: Optional[str]) -> Optional[float]:
    if not height_str or height_str == '0':
        return None

    # Handle formats like 5'6" or 5'6
    import re
    match = re.match(r"(\d+)'(\d*)", height_str)
    if match:
        feet = int(match.group(1) or 0)
        inches = int(match.group(2) or 0)
        result = feet + (inches / 12.0)
        return round(result, 1)

    try:
        parsed = float(height_str)
    except ValueError:
        return None

    if parsed == 0.0:
        return None
    return round(parsed, 1)


def _convert_weight_to_decimal(weight_str: Optional[str]) -> Optional[float]:
    if not weight_str or weight_str == '0':
        return None
    try:
        parsed = float(weight_str)
    except ValueError:
        return None
    return None if parsed == 0.0 else parsed


def _format_datetime(dt: datetime) -> str:
    return dt.isoformat()


def _format_datetime_string(date_str: Optional[str]) -> str:
    if not date_str:
        return _format_datetime(datetime.now())
    try:
        parsed = datetime.fromisoformat(date_str)
        return _format_datetime(parsed)
    except ValueError:
        return _format_datetime(datetime.now())


def _save_image_to_file(base64_image: str, treatment_id: str) -> str:
    try:
        treatment_dir = UPLOADS_DIR / 'Treatment' / treatment_id / 'Images'
        treatment_dir.mkdir(parents=True, exist_ok=True)

        image_id = f"{int(datetime.now().timestamp() * 1000)}-{datetime.now().microsecond}"
        filename = f'{image_id}.png'
        file_path = treatment_dir / filename

        base64_data = base64_image.split(',')[-1] if ',' in base64_image else base64_image
        image_bytes = base64.b64decode(base64_data)
        with open(file_path, 'wb') as f:
            f.write(image_bytes)

        return f'/Uploads/Treatment/{treatment_id}/Images/{filename}'
    except Exception as e:
        print(f'❌ Error saving image: {e}')
        return ''


def _get_dashboard_stats(conn: sqlite3.Connection, start: Optional[datetime], end: Optional[datetime]) -> Dict[str, Any]:
    where_clause = '(IsDeleted = 0 OR IsDeleted IS NULL)'
    args: list = []
    if start and end:
        where_clause += ' AND TreatmentDate >= ? AND TreatmentDate < ?'
        args.extend([start.isoformat(), end.isoformat()])

    patient_result = conn.execute(
        f'SELECT COUNT(*) as count FROM Treatment WHERE {where_clause}',
        args
    ).fetchone()

    collection_result = conn.execute(
        f'SELECT IFNULL(SUM(PaidFees), 0) as total FROM Treatment WHERE {where_clause}',
        args
    ).fetchone()

    return {
        'patients': patient_result['count'] if patient_result else 0,
        'collection': collection_result['total'] if collection_result else 0.0,
    }


@app.route('/api/search-reports', methods=['POST'])
def search_reports():
    data = request.json or {}
    query = (data.get('query') or '').lower()

    conn = get_db()
    c = conn.cursor()

    try:
        c.execute('''
            SELECT DISTINCT 
                t.Reports as details,
                t.Lab as lab,
                COUNT(*) as usage_count
            FROM Treatment t
            WHERE t.Reports IS NOT NULL 
            AND t.Reports != '' 
            AND t.Reports NOT LIKE '%{%' 
            AND t.Reports NOT LIKE '%}%' 
            AND t.Reports NOT LIKE '%"%'
            AND LENGTH(t.Reports) < 500
            GROUP BY t.Reports, t.Lab
            ORDER BY usage_count DESC
        ''')

        unique_reports: Dict[str, Dict[str, Any]] = {}

        for row in c.fetchall():
            reports_text = row['details'] or ''
            lab = row['lab'] or ''

            if '{' in reports_text or '}' in reports_text or '"' in reports_text:
                continue

            for report in [r.strip() for r in reports_text.split(',') if r.strip()]:
                if (
                    len(report) < 3 or
                    report.lower() in ['ok', 'aa', 'aaa', 'ahm', 'gg', 'raw', 'ffff', 'wer', 'sgpt', 'opop', 'huiha', 'jiojio', 'sara', 'sasas'] or
                    report.isdigit()
                ):
                    continue

                report_lower = report.lower()
                if not query or query in report_lower or query in lab.lower():
                    if report_lower not in unique_reports or unique_reports[report_lower]['usage_count'] < row['usage_count']:
                        unique_reports[report_lower] = {
                            'details': report,
                            'lab': lab,
                            'usage_count': row['usage_count']
                        }

        result_list = list(unique_reports.values())
        result_list.sort(key=lambda x: (-x['usage_count'], x['details'].lower()))

        return jsonify(result_list[:50])
    except Exception as e:
        print(f'Error searching reports: {e}')
        return jsonify([])
    finally:
        conn.close()


@app.route('/sync/pull', methods=['GET'])
def sync_pull():
    conn = get_db()
    c = conn.cursor()

    patients: Dict[str, Dict[str, Any]] = {}

    c.execute('SELECT * FROM Patient WHERE IsDeleted = 0 OR IsDeleted IS NULL')
    for row in c.fetchall():
        patient_id = row['OfflineId'] or str(row['Id'])
        patients[patient_id] = {
            'name': row['Name'] or '',
            'phone': row['Contact'] or '',
            'email': row['Email'],
            'age': row['Age'],
            'gender': row['Gender'],
            'visits': []
        }

    c.execute('''
        SELECT OfflineId, PatientOfflineId, PatientId, TreatmentDate, DieasesName, SymptomName,
               Lab, Remarks, ReferredDoctorName, ReferredHospitalName, Counsultancyfee, PaidFees,
               FollowUpDate, Height, Weight, PatientBP, NoOfPulse, SpoTwo, Temprature
        FROM Treatment
        WHERE IsDeleted = 0 OR IsDeleted IS NULL
    ''')

    treatments = c.fetchall()
    for treatment in treatments:
        try:
            patient_id = treatment['PatientOfflineId'] or str(treatment['PatientId'])
            treatment_id = treatment['OfflineId']

            c.execute('''
                SELECT medicine_name, Time, Type, MorningQuantity, AfternoonQuantity, EveningQuantity, Quantity
                FROM Prescription
                WHERE TreatmentOfflineId = ? AND (IsDeleted = 0 OR IsDeleted IS NULL)
            ''', (treatment_id,))

            medicines = []
            for presc in c.fetchall():
                dosage = f"{int(presc['MorningQuantity'] or 0)}-{int(presc['AfternoonQuantity'] or 0)}-{int(presc['EveningQuantity'] or 0)}"
                medicines.append({
                    'medicine': presc['medicine_name'] or '',
                    'dosage': dosage,
                    'meal': presc['Time'] or '',
                    'type': presc['Type'] or 'Tablet',
                    'quantity': presc['Quantity'] or ''
                })

            c.execute('''
                SELECT Image_Path FROM TreatmentImages
                WHERE TreatmentOfflineId = ? AND (IsDeleted = 0 OR IsDeleted IS NULL)
            ''', (treatment_id,))
            images = [row['Image_Path'] for row in c.fetchall()]

            referred = {
                'doctorName': treatment['ReferredDoctorName'] or '',
                'hospitalName': treatment['ReferredHospitalName'] or ''
            }

            total_fee = treatment['Counsultancyfee'] or 0
            paid_fees = treatment['PaidFees'] or 0

            visit = {
                'date': treatment['TreatmentDate'] or _format_datetime(datetime.now()),
                'issue': treatment['DieasesName'] or treatment['SymptomName'] or '',
                'symptoms': [],
                'diseases': [],
                'medicines': medicines,
                'examination': treatment['Lab'] or '',
                'remarks': treatment['Remarks'] or '',
                'report': {
                    'reportNames': [],
                    'height': str(treatment['Height'] or ''),
                    'weight': str(treatment['Weight'] or ''),
                    'bp': treatment['PatientBP'] or '',
                    'pulse': treatment['NoOfPulse'] or '',
                    'spo2': treatment['SpoTwo'] or '',
                    'temperature': treatment['Temprature'] or '',
                    'labCity': treatment['Lab'] or ''
                },
                'referred': referred,
                'images': images,
                'totalFee': total_fee,
                'amountPaid': paid_fees,
                'paid': paid_fees,
                'paidFee': paid_fees,
                'pendingAmount': (total_fee or 0) - (paid_fees or 0),
                'followUpDate': treatment['FollowUpDate']
            }

            if patient_id in patients:
                patients[patient_id]['visits'].append(visit)
        except Exception as e:
            print(f'Error processing treatment {treatment.get("OfflineId")}: {e}')
            continue

    conn.close()
    return jsonify(patients)


@app.route('/sync/push', methods=['POST'])
def sync_push():
    try:
        data = request.get_json(force=True) if not request.is_json else request.json
        if not isinstance(data, dict):
            return jsonify({'status': 'error', 'message': 'Invalid data format - expected JSON object'}), 400
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'JSON parsing error: {str(e)}'}), 400

    conn = get_db()

    try:
        patient_id = data.get('patientId') or ''
        patient_name = data.get('patientName') or ''
        patient_phone = data.get('patientPhone') or ''

        if not patient_id or not patient_name or not patient_phone:
            return jsonify({'status': 'error', 'message': 'Missing required patient data'}), 400

        existing_patient = conn.execute(
            'SELECT Id FROM Patient WHERE OfflineId = ? LIMIT 1',
            (patient_id,)
        ).fetchone()

        patient_db_id: Optional[int] = None
        if existing_patient:
            patient_db_id = existing_patient['Id']
            if patient_db_id is None:
                patient_db_id = int(datetime.now().timestamp() * 1000)
                conn.execute(
                    'UPDATE Patient SET Id = ? WHERE OfflineId = ?',
                    (patient_db_id, patient_id)
                )
        else:
            patient_db_id = int(datetime.now().timestamp() * 1000)
            _insert_row(conn, 'Patient', {
                'OfflineId': patient_id,
                'Id': patient_db_id,
                'Name': patient_name,
                'Contact': patient_phone,
                'IsdCode': '91',
                'Email': data.get('patientEmail', ''),
                'Age': data.get('patientAge') or data.get('age'),
                'Gender': data.get('patientGender', ''),
                'Address': data.get('patientAddress', ''),
                'IsDeleted': 0,
                'IsSync': 1
            })

        treatment_id = data.get('treatmentId') or f"treatment-{int(datetime.now().timestamp() * 1000)}-{datetime.now().microsecond}"
        total_fee = float(data.get('totalFee') or data.get('Counsultancyfee') or 0)
        paid_fee = float(data.get('amountPaid') or data.get('PaidFees') or 0)

        server_timestamp = _format_datetime(datetime.now())

        diseases = data.get('diseases') or data.get('DieasesName') or []
        if isinstance(diseases, str):
            diseases = [d.strip() for d in diseases.split(',') if d.strip()]

        symptoms = data.get('symptoms') or data.get('SymptomName') or []
        if isinstance(symptoms, str):
            symptoms = [s.strip() for s in symptoms.split(',') if s.strip()]

        reports_value = data.get('Reports')
        if reports_value is None:
            lab_reports = data.get('labReports') or {}
            reports_list = lab_reports.get('reports') or []
            if isinstance(reports_list, list):
                reports_value = ', '.join([r for r in reports_list if r])
            else:
                reports_value = ''

        treatment_data = {
            'OfflineId': treatment_id,
            'Id': int(datetime.now().timestamp() * 1000),
            'PatientOfflineId': patient_id,
            'PatientId': patient_db_id,
            'TreatmentDate': _format_datetime_string(data.get('date')),
            'DieasesName': ', '.join(diseases),
            'SymptomName': ', '.join(symptoms),
            'Counsultancyfee': total_fee,
            'PaidFees': paid_fee,
            'FollowUpDate': _format_datetime_string(data.get('followUpDate') or data.get('FollowUpDate')),
            'Remarks': data.get('Remarks') or data.get('examination') or data.get('remarks') or '',
            'Lab': data.get('Lab') or (data.get('labReports') or {}).get('labCity') or '',
            'Reports': reports_value or '',
            'Height': _convert_height_to_decimal(str(data.get('Height') or (data.get('patientDetails') or {}).get('height') or '')),
            'Weight': _convert_weight_to_decimal(str(data.get('Weight') or (data.get('patientDetails') or {}).get('weight') or '')),
            'PatientBP': data.get('PatientBP') or (data.get('patientDetails') or {}).get('bp') or '',
            'NoOfPulse': data.get('NoOfPulse') or (data.get('patientDetails') or {}).get('pulse') or '',
            'SpoTwo': data.get('SpoTwo') or (data.get('patientDetails') or {}).get('spo2') or '',
            'Temprature': data.get('Temprature') or (data.get('patientDetails') or {}).get('temperature') or '',
            'ReferredDoctorName': data.get('ReferredDoctorName') or (data.get('referred') or {}).get('doctorName') or '',
            'ReferredHospitalName': data.get('ReferredHospitalName') or (data.get('referred') or {}).get('clinicName') or '',
            'Photo': '',
            'IsInQueued': 0,
            'QueueId': None,
            'Rating': None,
            'Feedback': None,
            'CreatedOn': server_timestamp,
            'IsDeleted': 0,
            'IsSync': 1
        }

        _insert_row(conn, 'Treatment', treatment_data, conflict='REPLACE')

        medicines = data.get('medicines') or []
        for idx, med in enumerate(medicines):
            dosage = (med.get('dosage') or '').split('-') if isinstance(med, dict) else []
            morning = float(dosage[0]) if len(dosage) > 0 and dosage[0] else 0
            afternoon = float(dosage[1]) if len(dosage) > 1 and dosage[1] else 0
            evening = float(dosage[2]) if len(dosage) > 2 and dosage[2] else 0
            night = float(dosage[3]) if len(dosage) > 3 and dosage[3] else 0

            _insert_row(conn, 'Prescription', {
                'OfflineId': f"presc-{int(datetime.now().timestamp() * 1000)}-{idx}",
                'Id': int(datetime.now().timestamp() * 1000) + idx,
                'TreatmentOfflineId': treatment_id,
                'TreatmentId': patient_db_id,
                'PatientId': patient_db_id,
                'medicine_name': med.get('medicine') or med.get('name') if isinstance(med, dict) else '',
                'note': med.get('instructions') or med.get('meal') if isinstance(med, dict) else '',
                'Time': med.get('meal') or med.get('instructions') if isinstance(med, dict) else '',
                'Type': med.get('type') if isinstance(med, dict) else 'Tablet',
                'MorningQuantity': morning,
                'AfternoonQuantity': afternoon,
                'EveningQuantity': evening,
                'NightQuantity': night,
                'DurationDays': int(med.get('days') or 0) if isinstance(med, dict) else 0,
                'Quantity': med.get('quantity') if isinstance(med, dict) else '',
                'MedicineId': None,
                'IsDeleted': 0,
                'IsSync': 1
            }, conflict='IGNORE')

        images = data.get('images') or []
        for idx, image_item in enumerate(images):
            if isinstance(image_item, dict):
                base64_image = image_item.get('data') or ''
            else:
                base64_image = image_item

            if not base64_image:
                continue

            image_path = _save_image_to_file(base64_image, treatment_id)
            _insert_row(conn, 'TreatmentImages', {
                'OfflineId': f"img-{int(datetime.now().timestamp() * 1000)}-{idx}",
                'TreatmentOfflineId': treatment_id,
                'Id': int(datetime.now().timestamp() * 1000) + idx,
                'TreatmentId': patient_db_id,
                'Image_Path': image_path,
                'IsPrescriptionImage': 0,
                'IsDeleted': 0,
                'IsSync': 1,
                'ThumbnailImage': image_path,
            }, conflict='IGNORE')

        conn.commit()
        return jsonify({'status': 'success', 'message': 'Treatment saved successfully'})
    except Exception as e:
        conn.rollback()
        print(f'❌ Error saving treatment: {e}')
        return jsonify({'status': 'error', 'message': str(e)}), 500
    finally:
        conn.close()


@app.route('/add_patient', methods=['POST'])
def add_patient():
    data = request.json or {}
    conn = get_db()

    try:
        patient_id = data.get('patientId')
        name = data.get('name')
        phone = data.get('phone')

        if not patient_id or not name or not phone:
            return jsonify({'success': False, 'error': 'Missing required fields'}), 400

        existing = conn.execute(
            'SELECT 1 FROM Patient WHERE LOWER(Name) = ? AND Contact = ? AND (IsDeleted = 0 OR IsDeleted IS NULL) LIMIT 1',
            (name.lower(), phone)
        ).fetchone()

        if existing:
            return jsonify({'success': False, 'message': f'Patient {name} already exists'})

        _insert_row(conn, 'Patient', {
            'OfflineId': patient_id,
            'Id': int(datetime.now().timestamp() * 1000),
            'Name': name,
            'Contact': phone,
            'IsdCode': '91',
            'IsDeleted': 0,
            'IsSync': 1
        })

        conn.commit()
        return jsonify({'success': True, 'message': 'Patient added'})
    except Exception as e:
        conn.rollback()
        return jsonify({'success': False, 'error': str(e)}), 500
    finally:
        conn.close()


@app.route('/patient/<patient_id>/balance', methods=['GET'])
def get_patient_balance(patient_id: str):
    conn = get_db()

    try:
        rows = conn.execute(
            'SELECT Counsultancyfee, PaidFees FROM Treatment WHERE (PatientOfflineId = ? OR PatientId = ?) AND (IsDeleted = 0 OR IsDeleted IS NULL)',
            (patient_id, patient_id)
        ).fetchall()

        total_due = 0.0
        total_paid = 0.0

        for row in rows:
            total_due += float(row['Counsultancyfee'] or 0)
            total_paid += float(row['PaidFees'] or 0)

        balance = total_due - total_paid

        return jsonify({
            'patientId': patient_id,
            'totalDue': total_due,
            'totalPaid': total_paid,
            'balance': balance,
            'status': 'due' if balance > 0 else 'advance' if balance < 0 else 'clear'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        conn.close()


@app.route('/doctor/profile', methods=['GET'])
def get_doctor_profile():
    conn = get_db()

    try:
        row = conn.execute('SELECT Name, HospitalName FROM DoctorProfile LIMIT 1').fetchone()
        if row:
            doctor_name = row['Name'] or 'Doctor'
            hospital_name = row['HospitalName'] or ''
            greeting_name = doctor_name[4:] if doctor_name.startswith('Dr. ') else doctor_name
            return jsonify({
                'name': greeting_name,
                'hospitalName': hospital_name,
                'displayName': doctor_name
            })

        return jsonify({
            'name': 'Doctor',
            'hospitalName': '',
            'displayName': 'Doctor Diary'
        })
    except Exception:
        return jsonify({
            'name': 'Doctor',
            'hospitalName': '',
            'displayName': 'Doctor Diary'
        })
    finally:
        conn.close()


@app.route('/symptom/search', methods=['GET'])
def search_symptoms():
    query = (request.args.get('q') or '').lower()
    conn = get_db()

    try:
        if not query:
            rows = conn.execute('SELECT name FROM Symptoms ORDER BY name LIMIT 50').fetchall()
        else:
            rows = conn.execute(
                'SELECT name FROM Symptoms WHERE LOWER(name) LIKE ? ORDER BY name LIMIT 20',
                (f'%{query}%',)
            ).fetchall()

        return jsonify([{'name': row['name']} for row in rows])
    except Exception:
        return jsonify([])
    finally:
        conn.close()


@app.route('/disease/search', methods=['GET'])
def search_diseases():
    query = (request.args.get('q') or '').lower()
    conn = get_db()

    try:
        if not query:
            rows = conn.execute('SELECT name FROM Diseases ORDER BY name LIMIT 50').fetchall()
        else:
            rows = conn.execute(
                'SELECT name FROM Diseases WHERE LOWER(name) LIKE ? ORDER BY name LIMIT 20',
                (f'%{query}%',)
            ).fetchall()

        return jsonify([{'name': row['name']} for row in rows])
    except Exception:
        return jsonify([])
    finally:
        conn.close()


@app.route('/medicine/search', methods=['GET'])
def search_medicines():
    query = (request.args.get('q') or '').lower()
    conn = get_db()

    try:
        if not query:
            rows = conn.execute(
                "SELECT Medicine, Type FROM Medicines WHERE Medicine IS NOT NULL AND Medicine != '' ORDER BY Medicine LIMIT 50"
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT Medicine, Type FROM Medicines WHERE LOWER(Medicine) LIKE ? AND Medicine IS NOT NULL AND Medicine != '' ORDER BY Medicine LIMIT 20",
                (f'%{query}%',)
            ).fetchall()

        return jsonify([{'name': row['Medicine'], 'type': row['Type'] or 'Tablet'} for row in rows])
    except Exception:
        return jsonify([])
    finally:
        conn.close()


@app.route('/api/consultancy-fees', methods=['GET'])
def consultancy_fees():
    conn = get_db()

    try:
        row = conn.execute('SELECT ConsultancyFees FROM DoctorProfile LIMIT 1').fetchone()
        fees = row['ConsultancyFees'] if row else 0
        return jsonify({'consultancyFees': fees})
    except Exception:
        return jsonify({'consultancyFees': 0})
    finally:
        conn.close()


@app.route('/api/dashboard-stats', methods=['GET'])
def dashboard_stats():
    conn = get_db()

    try:
        period = request.args.get('period', 'total')
        start_date = request.args.get('start')
        end_date = request.args.get('end')

        now = datetime.now()

        if period == 'today':
            start = datetime(now.year, now.month, now.day)
            end = start + timedelta(days=1)
        elif period == 'week':
            start = datetime(now.year, now.month, now.day) - timedelta(days=6)
            end = datetime(now.year, now.month, now.day) + timedelta(days=1)
        elif period == 'month':
            start = datetime(now.year, now.month, 1)
            if now.month == 12:
                end = datetime(now.year + 1, 1, 1)
            else:
                end = datetime(now.year, now.month + 1, 1)
        elif period == 'custom':
            if start_date and end_date:
                start = datetime.fromisoformat(start_date)
                end = datetime.fromisoformat(end_date) + timedelta(days=1)
            else:
                raise ValueError('Custom period requires start and end dates')
        else:
            stats = _get_dashboard_stats(conn, None, None)
            return jsonify(stats)

        stats = _get_dashboard_stats(conn, start, end)
        return jsonify(stats)
    except Exception as e:
        return jsonify({'patients': 0, 'collection': 0.0, 'error': str(e)}), 500
    finally:
        conn.close()


@app.route('/api/test/fees', methods=['GET'])
def test_fees():
    return jsonify({'fees': 999, 'message': 'Test endpoint'})


@app.route('/api/nlp/extract', methods=['POST'])
def nlp_extract():
    global _LATEST_NLP
    data = request.json or {}
    text = (data.get('text') or '').strip()
    stage = data.get('stage')
    language = data.get('language')

    if not text:
        return jsonify({'error': 'Missing transcript text'}), 400

    print(f"🧠 NLP: /api/nlp/extract received (stage={stage}, language={language})")
    extracted = _groq_extract_medical(text, stage=stage, language=language)
    payload = {
        'timestamp': time.time(),
        'data': extracted,
        'text': text,
        'stage': stage,
        'language': language
    }
    _LATEST_NLP = payload
    print("✅ NLP: /api/nlp/extract stored latest payload")
    return jsonify(payload)


@app.route('/api/nlp/latest', methods=['GET'])
def nlp_latest():
    since_raw = request.args.get('since', '0')
    try:
        since = float(since_raw)
    except ValueError:
        since = 0.0

    if _LATEST_NLP.get('timestamp', 0) <= since:
        return jsonify({'timestamp': _LATEST_NLP.get('timestamp', 0), 'data': None})

    return jsonify(_LATEST_NLP)


@app.route('/api/stt/start', methods=['POST'])
def stt_start():
    if sr is None:
        return jsonify({"status": "error", "message": "speech_recognition not installed"}), 500
    success = _STT_HANDLER.start_listening()
    return jsonify({"status": "success" if success else "error", "stt": _STT_STATUS})


@app.route('/api/stt/stop', methods=['POST'])
def stt_stop():
    _STT_HANDLER.stop_listening_service()
    return jsonify({"status": "success", "stt": _STT_STATUS})


@app.route('/api/stt/status', methods=['GET'])
def stt_status():
    return jsonify(_STT_STATUS)


@app.route('/')
def index():
    return send_from_directory(str(BASE_DIR), 'index_mobile.html')


@app.route('/favicon.ico')
def favicon():
    """Serve the DoctorDiary logo for browsers that request the conventional favicon URL."""
    return send_file(BASE_DIR / 'favicon.png', mimetype='image/png', max_age=0)


@app.route('/index.html')
def index_html():
    return send_from_directory(str(BASE_DIR), 'index_mobile.html')


# Voice Integration Endpoints
@app.route('/voice/start', methods=['POST'])
def voice_start():
    global _VOICE_INTEGRATION, _VOICE_CLIENTS
    
    print('🎤 Voice start endpoint called')
    data = request.json or {}
    session_id = data.get('session_id', 'default')
    print(f'📋 Session ID: {session_id}')
    
    if not _VOICE_INTEGRATION:
        # Initialize on first use
        config = data.get('config', {})
        project_id = config.get('project_id') or os.environ.get('GOOGLE_CLOUD_PROJECT')
        credentials_path = config.get('credentials_path') or os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')
        openai_key = config.get('openai_key') or os.environ.get('OPENAI_API_KEY')
        
        print(f'🔧 Config - Project: {project_id}, Creds: {credentials_path}, OpenAI: {"set" if openai_key else "missing"}')
        
        if not all([project_id, credentials_path, openai_key]):
            print('❌ Missing voice configuration')
            return jsonify({'status': 'error', 'message': 'Missing voice configuration'}), 400
        
        try:
            print('🔄 Initializing VoiceIntegration...')
            _VOICE_INTEGRATION = VoiceIntegration(project_id, credentials_path, openai_key)
            print('✅ VoiceIntegration initialized')
        except Exception as e:
            print(f'❌ Failed to initialize voice: {e}')
            return jsonify({'status': 'error', 'message': f'Failed to initialize voice: {str(e)}'}), 500
    
    if session_id in _VOICE_CLIENTS:
        print('⚠️ Session already active')
        return jsonify({'status': 'error', 'message': 'Session already active'}), 400
    
    # Create callback to store data
    session_data = {'transcripts': [], 'entities': {}}
    _VOICE_CLIENTS[session_id] = session_data
    
    def callback(data):
        print(f'📞 Callback received: {data.get("type")}')
        if data['type'] == 'transcript':
            print(f'📝 Transcript: {data.get("text")}')
            session_data['transcripts'].append(data)
        elif data['type'] == 'entities':
            print(f'🏷️ Entities: {data.get("data")}')
            session_data['entities'] = data['data']
    
    print('🎙️ Starting listening...')
    _VOICE_INTEGRATION.start_listening(callback)
    print('✅ Voice listening started')
    
    return jsonify({'status': 'success', 'message': 'Voice listening started', 'session_id': session_id})


@app.route('/voice/stop', methods=['POST'])
def voice_stop():
    global _VOICE_INTEGRATION, _VOICE_CLIENTS
    
    data = request.json or {}
    session_id = data.get('session_id', 'default')
    
    if _VOICE_INTEGRATION:
        _VOICE_INTEGRATION.stop_listening()
    
    session_data = _VOICE_CLIENTS.pop(session_id, None)
    
    return jsonify({
        'status': 'success',
        'message': 'Voice listening stopped',
        'data': session_data or {}
    })


@app.route('/voice/status', methods=['GET'])
def voice_status():
    session_id = request.args.get('session_id', 'default')
    session_data = _VOICE_CLIENTS.get(session_id, {})
    
    return jsonify({
        'status': 'success',
        'active': session_id in _VOICE_CLIENTS,
        'data': session_data
    })


@app.route('/<path:filename>')
def static_files(filename: str):
    return send_from_directory(str(BASE_DIR), filename)


if __name__ == '__main__':
    web_host = os.getenv('DD_HOST', '127.0.0.1')
    web_port = int(os.getenv('DD_PORT', '8080'))

    print('🚀 Unified Server Starting...')
    print(f'📊 SQLite: {DB_PATH}')
    print(f'🌐 Web App: http://localhost:{web_port}')
    print(f'📱 Open: http://localhost:{web_port}/index.html')
    print(f'🔗 API: http://localhost:{web_port}/sync/pull')
    print('Press Ctrl+C to stop')

    app.run(host=web_host, port=web_port, debug=False)
