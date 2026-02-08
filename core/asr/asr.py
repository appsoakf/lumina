# import json
# import os
# import wave
# import pyaudio
# import sherpa_onnx
# import numpy as np
# import soundfile as sf



# class asr:
#     def __init__(self):
#         self.asr_model_path = "data/model/ASR/sherpa-onnx-sense-voice-zh-en-ja-ko-yue"
#         self.vp_model_path = "data/model/SpeakerID/3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx"
#         vp_config, extractor, audio1, sample_rate1, embedding1 = None, None, None, None, None
#         with open('data/config/config.json', 'r', encoding='utf-8') as f:
#             config = json.load(f)
#         self.asr_sensitivity = config["other_settings"]["语音识别灵敏度"]
#         self.voiceprint_switch = config["feature_switches"]["声纹识别"]
        
#         self.mic_num = int(config["other_settings"]["麦克风编号"])
#         self.voiceprint_threshold = float(config["feature_switches"]["声纹识别阈值"])
#         silence_duration_map = {"高": 1, "中": 2, "低": 3}
#         self.silence_duration = silence_duration_map.get(self.asr_sensitivity, 3)
#         self.audio_format = pyaudio.paInt16

#         # channels表示声道，asr模型通常采用单声道
#         # rate表示一秒采样的帧数，与SenseVoice模型训练采样率一致
#         # chunk表示每次回调处理的帧数。用于控制静音的时间精度。chunk越大，则检测静音越迟钝，导致最后包含较长一段静音
#         #   chunk越小，回调粒度越细，能够控制静音片段很少
#         # 这样设置是为了与Sherpa-onnx模型对齐
#         self.channels, self.rate, self.chunk = 1, 16000, 1024
#         self.silence_chunks = self.silence_duration * self.rate / self.chunk # 静音持续的帧数
#         self.p, self.stream, self.recognizer = None, None, None
#         self.cache_path = "data/cache/cache_record.wav"
#         self.model = f"{self.asr_model_path}/model.int8.onnx"
#         self.tokens = f"{self.asr_model_path}/tokens.txt"

#     def _rms(data):  # 计算音频数据的均方根
#         return np.sqrt(np.mean(np.frombuffer(data, dtype=np.int16) ** 2))


#     def _dbfs(rms_value):  # 将均方根转换为分贝满量程（dBFS）
#         return 20 * np.log10(rms_value / (2 ** 15))  # 16位音频
    
#     # 录音
#     def record_audio(self):  
#         frames = []
#         recording = True
#         silence_counter = 0  # 用于记录静音持续的帧数
#         if self.p is None:
#             self.p = pyaudio.PyAudio()
#             # 创建音频流，捕捉麦克风输入
#             stream = self.p.open(format=self.format, channels=self.channels, rate=self.rate, 
#                             input=True, frames_per_buffer=self.chunk,
#                             input_device_index=self.mic_num)
#         while recording:
#             data = stream.read(self.chunk)
#             frames.append(data)
#             current_rms = self._rms(data)
#             current_dbfs = self._dbfs(current_rms)

#             if str(current_dbfs) != "nan":
#                 silence_counter += 1  # 增加静音计数
#                 if silence_counter > self.silence_chunks:  # 判断是否达到设定的静音持续时间
#                     recording = False
#             else:
#                 silence_counter = 0  # 重置静音计数
#         return b''.join(frames) # 返回完整的原始音频PCM数据字节串（bytes对象
    


#     def recognize_audio(self, audiodata):  # 语音识别
#         if self.recognizer is None:
#             self.recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(model=self.model, tokens=self.tokens, use_itn=True,
#                                                                         num_threads=int(os.cpu_count()) - 1)
#         with wave.open(self.cache_path, 'wb') as wf:
#             wf.setnchannels(self.channels)
#             wf.setsampwidth(self.p.get_sample_size(self.format))
#             wf.setframerate(self.rate)
#             wf.writeframes(audiodata)
#         with wave.open(self.cache_path, 'rb') as wf:
#             n_frames = wf.getnframes()
#             duration = n_frames / self.rate
#         if duration < self.silence_duration + 0.5:
#             return ""
#         # if voiceprint_switch == "开启":
#         #     if not verify_speakers():
#         #         return ""
#         audio, sample_rate = sf.read(self.cache_path, dtype="float32", always_2d=True)
#         asr_stream = self.recognizer.create_stream()
#         asr_stream.accept_waveform(sample_rate, audio[:, 0])
#         self.recognizer.decode_stream(asr_stream)
#         res = json.loads(str(asr_stream.result))
#         emotion_key = res.get('emotion', '').strip('<|>')
#         event_key = res.get('event', '').strip('<|>')
#         text = res.get('text', '')
#         emotion_dict = {"HAPPY": "[开心]", "SAD": "[伤心]", "ANGRY": "[愤怒]", "DISGUSTED": "[厌恶]", "SURPRISED": "[惊讶]",
#                         "NEUTRAL": "", "EMO_UNKNOWN": ""}
#         event_dict = {"BGM": "", "Applause": "[鼓掌]", "Laughter": "[大笑]", "Cry": "[哭]", "Sneeze": "[打喷嚏]",
#                     "Cough": "[咳嗽]", "Breath": "[深呼吸]", "Speech": "", "Event_UNK": ""}
#         emotion = emotion_dict.get(emotion_key, "")
#         event = event_dict.get(event_key, "")
#         result = event + text + emotion
#         if result == "The.":
#             return ""
#         return result