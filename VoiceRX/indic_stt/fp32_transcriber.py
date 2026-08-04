"""
Full-precision (fp32) IndicConformer transcriber.

The `indic-asr-onnx` package only serves the int8-quantized model, whose WER is
~2x worse than full precision (poor on non-Hindi languages). This class runs the
FULL-PRECISION model from an ungated mirror
(`sunilmahendrakar/indic-conformer-600m-multilingual`), reusing the same decoding
logic but pointed at the fp32 assets. Supports both CTC and RNNT decoding.
"""
import json
import os
import tempfile
import uuid

import numpy as np
import onnxruntime as ort
import torch
import torchaudio
from huggingface_hub import snapshot_download

DEFAULT_FP32_REPO = os.getenv(
    "INDIC_FP32_REPO", "sunilmahendrakar/indic-conformer-600m-multilingual"
)
BLANK_ID = 256
CHUNK_SECONDS = float(os.getenv("INDIC_CHUNK_SECONDS", "20"))
CHUNK_OVERLAP_SECONDS = float(os.getenv("INDIC_CHUNK_OVERLAP_SECONDS", "1"))

# RNNT decoder/joint-network constants, matching the repo's IndicASRConfig defaults.
SOS = 5632
PRED_RNN_LAYERS = 2
PRED_RNN_HIDDEN_DIM = 640
RNNT_MAX_SYMBOLS = 10
FRAME_DURATION_MS = 0.04
# RNNT decoding carries recurrent decoder state across chunks (see transcribe_rnnt),
# so only the newly-seen encoder frames of each chunk need decoding.
OVERLAP_FRAMES = round(CHUNK_OVERLAP_SECONDS / FRAME_DURATION_MS)


class Fp32IndicTranscriber:
    def __init__(self, repo_id: str = DEFAULT_FP32_REPO, model_dir: str | None = None):
        self.model_dir = model_dir or snapshot_download(repo_id=repo_id)
        self.providers = ["CPUExecutionProvider"]
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000, n_fft=512, win_length=400, hop_length=160,
            f_min=0.0, f_max=8000.0, n_mels=80, window_fn=torch.hann_window, power=2.0,
        )
        self._encoder = None
        self._ctc = None
        self._rnnt = None
        self._joint_enc = None
        self._joint_pred = None
        self._joint_pre_net = None
        self._joint_post_net = {}
        with open(self._asset("vocab.json"), encoding="utf-8") as f:
            self._vocab_all = json.load(f)
        with open(self._asset("language_masks.json"), encoding="utf-8") as f:
            self._masks_all = json.load(f)

    def _asset(self, name: str) -> str:
        return os.path.join(self.model_dir, "assets", name)

    def _ensure_encoder(self):
        if self._encoder is None:
            self._encoder = ort.InferenceSession(self._asset("encoder.onnx"), providers=self.providers)

    def _ensure_ctc(self):
        self._ensure_encoder()
        if self._ctc is None:
            self._ctc = ort.InferenceSession(self._asset("ctc_decoder.onnx"), providers=self.providers)

    def _ensure_rnnt(self):
        self._ensure_encoder()
        if self._rnnt is None:
            self._rnnt = ort.InferenceSession(self._asset("rnnt_decoder.onnx"), providers=self.providers)
        if self._joint_enc is None:
            self._joint_enc = ort.InferenceSession(self._asset("joint_enc.onnx"), providers=self.providers)
        if self._joint_pred is None:
            self._joint_pred = ort.InferenceSession(self._asset("joint_pred.onnx"), providers=self.providers)
        if self._joint_pre_net is None:
            self._joint_pre_net = ort.InferenceSession(self._asset("joint_pre_net.onnx"), providers=self.providers)

    def _ensure_joint_post_net(self, lang: str) -> ort.InferenceSession:
        if lang not in self._joint_post_net:
            self._joint_post_net[lang] = ort.InferenceSession(
                self._asset(f"joint_post_net_{lang}.onnx"), providers=self.providers
            )
        return self._joint_post_net[lang]

    def _mel_features(self, waveform: torch.Tensor) -> np.ndarray:
        feats = self.mel_transform(waveform)
        feats = torch.log(feats + 1e-9)
        mean = feats.mean(dim=2, keepdim=True)
        std = feats.std(dim=2, keepdim=True) + 1e-5
        feats = (feats - mean) / std
        return feats.squeeze(0).cpu().numpy().astype(np.float32)

    def _preprocess(self, audio_path: str) -> np.ndarray:
        waveform, sr = torchaudio.load(audio_path)
        if waveform.ndim > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != 16000:
            waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)(waveform)
        return self._mel_features(waveform)

    def _run_encoder(self, feats: np.ndarray):
        length = np.array([feats.shape[1]], dtype=np.int64)
        feats = np.expand_dims(feats, axis=0)
        ei = self._encoder.get_inputs()
        ed = {ei[0].name: feats}
        if len(ei) > 1:
            ed[ei[1].name] = length
        enc_out = self._encoder.run(None, ed)[0]
        return enc_out, length

    def _transcribe_chunk(self, audio_path: str, lang: str) -> str:
        vocab = self._vocab_all[lang]
        masks = self._masks_all[lang]
        self._ensure_ctc()

        feats = self._preprocess(audio_path)
        enc_out, length = self._run_encoder(feats)

        ci = self._ctc.get_inputs()
        cd = {ci[0].name: enc_out}
        if len(ci) > 1:
            cd[ci[1].name] = length
        logits = self._ctc.run(None, cd)[0]

        mask = np.array(masks, dtype=bool)
        logits_sliced = logits[:, :, mask]
        pred_ids = np.argmax(logits_sliced, axis=-1)[0]

        tokens = []
        prev = None
        for idx in pred_ids:
            if idx != prev and idx != BLANK_ID and idx < len(vocab):
                tokens.append(vocab[idx])
            prev = idx
        return "".join(tokens).replace("\u2581", " ").strip()

    def transcribe_ctc(self, audio_path: str, lang: str) -> str:
        """Transcribe in short overlapping windows for long consultations.

        IndicConformer is more reliable when each inference window is short.
        The Flask service still receives one uploaded recording; chunking is
        kept here so both the web and Flutter clients use the same behavior.
        """
        waveform, sample_rate = torchaudio.load(audio_path)
        if waveform.ndim > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sample_rate != 16000:
            waveform = torchaudio.transforms.Resample(
                orig_freq=sample_rate, new_freq=16000
            )(waveform)
            sample_rate = 16000

        total_samples = waveform.shape[-1]
        chunk_samples = max(1, int(CHUNK_SECONDS * sample_rate))
        overlap_samples = max(0, min(chunk_samples - 1, int(CHUNK_OVERLAP_SECONDS * sample_rate)))
        if total_samples <= chunk_samples:
            return self._transcribe_chunk(audio_path, lang)

        step_samples = chunk_samples - overlap_samples
        transcripts = []
        start = 0
        while start < total_samples:
            end = min(total_samples, start + chunk_samples)
            chunk = waveform[:, start:end]
            temp_path = os.path.join(tempfile.gettempdir(), f"indic_chunk_{uuid.uuid4().hex}.wav")
            try:
                torchaudio.save(temp_path, chunk, sample_rate)
                text = self._transcribe_chunk(temp_path, lang)
                if text:
                    transcripts.append(text)
            finally:
                try:
                    os.unlink(temp_path)
                except FileNotFoundError:
                    pass
            if end >= total_samples:
                break
            start += step_samples

        return " ".join(transcripts).strip()

    def transcribe_rnnt(self, audio_path: str, lang: str) -> str:
        """Chunked greedy RNNT decode for long consultations.

        The RNNT decoder is recurrent (carries hidden state across time steps),
        so unlike CTC, resetting it at each chunk boundary drops content there.
        Instead the decoder state and in-progress hypothesis are carried
        continuously across chunks; each chunk after the first only decodes its
        newly-seen encoder frames (OVERLAP_FRAMES already covered by the
        previous chunk's tail).
        """
        self._ensure_rnnt()
        joint_post_net = self._ensure_joint_post_net(lang)
        vocab = self._vocab_all[lang]

        waveform, sample_rate = torchaudio.load(audio_path)
        if waveform.ndim > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sample_rate != 16000:
            waveform = torchaudio.transforms.Resample(
                orig_freq=sample_rate, new_freq=16000
            )(waveform)
            sample_rate = 16000

        total_samples = waveform.shape[-1]
        chunk_samples = max(1, int(CHUNK_SECONDS * sample_rate))
        overlap_samples = max(0, min(chunk_samples - 1, int(CHUNK_OVERLAP_SECONDS * sample_rate)))
        step_samples = chunk_samples - overlap_samples

        hyp = [SOS]
        prev_dec_state = (
            np.zeros((PRED_RNN_LAYERS, 1, PRED_RNN_HIDDEN_DIM), dtype=np.float32),
            np.zeros((PRED_RNN_LAYERS, 1, PRED_RNN_HIDDEN_DIM), dtype=np.float32),
        )

        start = 0
        first_chunk = True
        while start < total_samples:
            end = min(total_samples, start + chunk_samples)
            feats = self._mel_features(waveform[:, start:end])
            enc_out, _ = self._run_encoder(feats)

            joint_enc = self._joint_enc.run(['output'], {'input': enc_out.transpose(0, 2, 1)})[0]
            joint_enc = torch.from_numpy(joint_enc)

            t_start = 0 if first_chunk else min(OVERLAP_FRAMES, joint_enc.size(1))
            first_chunk = False

            for t in range(t_start, joint_enc.size(1)):
                f = joint_enc[:, t, :].unsqueeze(1)
                not_blank = True
                symbols_added = 0
                while not_blank and symbols_added < RNNT_MAX_SYMBOLS:
                    g, _, dec_state_0, dec_state_1 = self._rnnt.run(
                        ['outputs', 'prednet_lengths', 'states', '162'],
                        {'targets': np.array([[hyp[-1]]], dtype=np.int32),
                         'target_length': np.array([1], dtype=np.int32),
                         'states.1': prev_dec_state[0],
                         'onnx::Slice_3': prev_dec_state[1]})
                    g = self._joint_pred.run(['output'], {'input': g.transpose(0, 2, 1)})[0]
                    joint_out = f + torch.from_numpy(g)
                    joint_out = self._joint_pre_net.run(['output'], {'input': joint_out.numpy()})[0]
                    logits = joint_post_net.run(['output'], {'input': joint_out})[0]
                    log_probs = torch.from_numpy(logits).log_softmax(dim=-1)
                    pred_token = log_probs.argmax(dim=-1).item()
                    if pred_token == BLANK_ID:
                        not_blank = False
                    else:
                        hyp.append(pred_token)
                        prev_dec_state = (dec_state_0, dec_state_1)
                    symbols_added += 1

            if end >= total_samples:
                break
            start += step_samples

        return "".join(vocab[x] for x in hyp if x != SOS).replace("▁", " ").strip()
