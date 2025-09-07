import pyaudio
import webrtcvad
import wave
import time
import collections
import numpy as np
from threading import Lock
import threading
from pathlib import Path
import os
import logging

class VoiceActivityDetector:
    def __init__(self, aggressiveness=3, sample_rate=16000, frame_duration_ms=30):
        """
        初始化人声活动检测器
        
        参数:
            aggressiveness: 检测激进程度 (0-3)
            sample_rate: 采样率 (Hz)
            frame_duration_ms: 帧持续时间 (ms)
        """
        self.vad = webrtcvad.Vad(aggressiveness)
        self.sample_rate = sample_rate
        self.frame_duration_ms = frame_duration_ms
        self.frame_size = int(sample_rate * frame_duration_ms / 1000)
        
    def is_speech(self, audio_frame):
        """检测单帧音频是否包含人声"""
        try:
            return self.vad.is_speech(audio_frame, self.sample_rate)
        except:
            return False

class VoiceWakeWord:
    def __init__(self, sample_rate=16000, channels=1, format=pyaudio.paInt16):
        """
        初始化语音唤醒系统
        
        参数:
            sample_rate: 采样率
            channels: 声道数
            format: 音频格式
        """
        self.sample_rate = sample_rate
        self.channels = channels
        self.format = format
        self.audio = pyaudio.PyAudio()
        self.vad = VoiceActivityDetector(aggressiveness=2)  # 降低VAD灵敏度
        self.is_listening = False
        self.audio_data = []
        self.data_lock = Lock()
        
        # 语音检测参数 - 调整为更严格
        self.silence_limit = 1.5  # 静音持续时间限制(秒)
        self.speech_limit = 0.43  # 语音持续时间限制(秒) - 提高要求
        self.pre_speech_buffer = 0.7  # 语音开始前缓冲时间(秒)
        self.post_speech_buffer = 1.8  # 语音结束后缓冲时间(秒) - 增加后缓冲
        
        # 计算帧数
        self.frame_duration = self.vad.frame_duration_ms / 1000.0
        self.pre_speech_frames = int(self.pre_speech_buffer / self.frame_duration)
        self.post_speech_frames = int(self.post_speech_buffer / self.frame_duration)
        
        # 状态变量
        self.silence_count = 0
        self.speech_count = 0
        self.consecutive_speech = 0  # 连续语音帧计数
        self.consecutive_silence = 0  # 连续静音帧计数
        self.recording = False
        self.in_speech = False
        self.speech_start_index = 0
        self.buffer = collections.deque(maxlen=self.pre_speech_frames)
        
        # 能量检测参数 - 调整为更严格
        self.energy_threshold = 1260  # 提高能量阈值
        self.min_energy_threshold = 360  # 最小能量阈值
        self.energy_window = 10  # 增加能量检测窗口
        self.energy_history = collections.deque(maxlen=50)  # 能量历史记录
        
        # 平滑检测参数
        self.speech_confirmation_frames = 3  # 需要连续几帧确认语音
        self.silence_confirmation_frames = 9  # 需要连续几帧确认静音
        
    def calculate_energy(self, audio_frame):
        """计算音频帧的能量"""
        try:
            audio_data = np.frombuffer(audio_frame, dtype=np.int16)
            # 使用绝对值的平均值，对突发噪声更鲁棒
            energy = np.mean(np.abs(audio_data.astype(np.float32)))
            return energy
        except:
            return 0
    
    def update_energy_threshold(self, current_energy):
        """动态更新能量阈值"""
        if len(self.energy_history) > 10:
            avg_background = np.mean(list(self.energy_history))
            # 阈值为基础噪声的2-3倍
            self.energy_threshold = max(self.min_energy_threshold, avg_background * 2.5)
        self.energy_history.append(current_energy)
    
    def is_reliable_speech(self, audio_frame):
        """更可靠的语音检测"""
        # VAD检测
        is_speech_vad = self.vad.is_speech(audio_frame)
        
        # 能量检测
        energy = self.calculate_energy(audio_frame)
        self.update_energy_threshold(energy)
        is_speech_energy = energy > self.energy_threshold
        
        # 需要VAD和能量检测同时为真才认为是可靠语音
        return is_speech_vad and is_speech_energy
    
    def start_listening(self, callback=None):
        """开始监听语音"""
        self.is_listening = True
        self.callback = callback
        
        # 打开音频流
        self.stream = self.audio.open(
            format=self.format,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.vad.frame_size,
            stream_callback=self._audio_callback
        )
        
        print("开始监听语音...")
        print(f"配置: 前缓冲{self.pre_speech_buffer}s, 后缓冲{self.post_speech_buffer}s")
        print(f"能量阈值: {self.energy_threshold}")
        self.stream.start_stream()
        
        # 保持主线程运行
        try:
            while self.stream.is_active() and self.is_listening:
                time.sleep(0.1)
        except KeyboardInterrupt:
            self.stop_listening()
        except Exception as e:
            print(f"监听错误: {e}")
            self.stop_listening()
    
    def stop_listening(self):
        """停止监听语音"""
        self.is_listening = False
        if hasattr(self, 'stream'):
            try:
                self.stream.stop_stream()
                self.stream.close()
            except:
                pass
        self.audio.terminate()
        print("停止监听语音")
    
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """音频回调函数"""
        if not self.is_listening:
            return (in_data, pyaudio.paContinue)
            
        with self.data_lock:
            # 始终保持缓冲区中有最新的pre_speech_frames帧数据
            self.buffer.append(in_data)
            
            # 使用更可靠的语音检测
            is_speech = self.is_reliable_speech(in_data)
            
            # 更新连续计数
            if is_speech:
                self.consecutive_speech += 1
                self.consecutive_silence = 0
                self.silence_count = 0  # 重置静音计数
            else:
                self.consecutive_silence += 1
                self.consecutive_speech = 0
            
            # 需要连续多帧确认才开始录音
            speech_confirmed = self.consecutive_speech >= self.speech_confirmation_frames
            silence_confirmed = self.consecutive_silence >= self.silence_confirmation_frames
            
            if speech_confirmed and not self.recording:
                # 开始录音，包含前缓冲区的数据
                print("检测到可靠语音活动，开始录音")
                self.recording = True
                self.in_speech = True
                self.speech_count = self.consecutive_speech
                self.audio_data = list(self.buffer)  # 包含前缓冲数据
                self.speech_start_index = len(self.audio_data) - self.consecutive_speech
            
            elif self.recording:
                # 继续录音
                self.audio_data.append(in_data)
                
                if is_speech:
                    self.speech_count += 1
                    self.silence_count = 0  # 重置静音计数
                else:
                    self.silence_count += 1
                
                # 只有在语音段中且检测到静音时才考虑结束录音
                if self.in_speech and not is_speech:
                    # 静音时间超过后缓冲时间，结束录音
                    if (self.silence_count * self.frame_duration >= 
                        self.post_speech_buffer):
                        print("语音活动结束，处理录音")
                        self._process_recording()
                
            return (in_data, pyaudio.paContinue)

    def _process_recording(self):
        """处理录音数据"""
        if not self.audio_data:
            self._reset_recording()
            return
            
        # 计算实际语音长度（从语音开始到录音结束）
        total_frames = len(self.audio_data)
        speech_duration = total_frames * self.frame_duration
        
        # 计算有效语音长度（从语音开始到语音结束）
        if self.in_speech:
            # 语音结束位置大约是总长度减去静音计数
            speech_end_index = max(self.speech_start_index, 
                                total_frames - int(self.silence_count * 0.8))
            effective_speech_frames = speech_end_index - self.speech_start_index
            effective_duration = effective_speech_frames * self.frame_duration
        else:
            effective_duration = 0
        
        print(f"总录音长度: {speech_duration:.2f}s, 有效语音长度: {effective_duration:.2f}s")
        
        if effective_duration >= self.speech_limit:
            print(f"触发回调")
            if self.callback:
                audio_buffer = b''.join(self.audio_data)
                threading.Thread(
                    target=self._execute_callback,
                    args=(audio_buffer,),
                    daemon=True
                ).start()
        else:
            print(f"语音过短，丢弃")
        
        self._reset_recording()
    
    def _execute_callback(self, audio_data):
        """执行回调函数"""
        try:
            if self.callback is not None and audio_data is not None:
                self.callback(audio_data)
        except Exception as e:
            print(f"回调函数执行错误: {e}")
     
    def _reset_recording(self):
        """重置录音状态"""
        self.recording = False
        self.in_speech = False
        self.speech_count = 0
        self.silence_count = 0
        self.consecutive_speech = 0
        self.consecutive_silence = 0
        self.speech_start_index = 0
        self.audio_data = []
    
    def save_audio(self, filename, audio_data):
        """保存音频数据到文件"""
        try:
            # 确保filename是字符串而不是Path对象
            filename_str = str(filename) if isinstance(filename, Path) else filename
            wf = wave.open(filename_str, 'wb')
            wf.setnchannels(self.channels)
            wf.setsampwidth(self.audio.get_sample_size(self.format))
            wf.setframerate(self.sample_rate)
            wf.writeframes(audio_data)
            wf.close()
            print(f"音频已保存到: {filename_str}")
        except Exception as e:
            print(f"保存音频失败: {e}")

class STTMethod():
    def __init__(self) -> None:
        self.model_path = Path(__file__).parent / "pretrained_models" / "SenseVoiceSmall"
        self.model = None
        pass

    def download_model(self):
        # SDK模型下载
        from modelscope import snapshot_download
        model_dir = snapshot_download('iic/SenseVoiceSmall', local_dir=str(self.model_path))
        if model_dir:
            pass
        else:
            raise FileNotFoundError
        
    def load_model(self):
        from funasr import AutoModel

        model_dir = self.model_path
        if os.path.exists(model_dir):
            pass
        else:
            logging.error("Model not find,downloading")
            self.download_model()

        model = AutoModel(
            model=str(model_dir),  # 确保传递字符串
            trust_remote_code=True,
            disable_update=True, 
            device="cuda:0",
        )
        return model
    """
            remote_code="./model.py",  
            vad_model="fsmn-vad",
            vad_kwargs={"max_single_segment_time": 30000},
    """
    
    def SenseVoice(self, wav_path):
        # 确保wav_path是字符串
        wav_path_str = str(wav_path) if isinstance(wav_path, Path) else wav_path
        
        from funasr.utils.postprocess_utils import rich_transcription_postprocess

        if self.model is None:
            self.model = self.load_model()
        # en
        res = self.model.generate(
            input=wav_path_str,  # 使用字符串路径
            cache={},
            language="auto",  # "zn", "en", "yue", "ja", "ko", "nospeech"
            use_itn=True,
            batch_size_s=60,
            merge_vad=True,  #
            merge_length_s=15,
            ban_emo_unk = False,
        )
        text = rich_transcription_postprocess(res[0]["text"])

        print(f"ASR got text:{text}")

        return text
    
def on_wake(audio_data):
    wav_path = Path(__file__).parent / "temp" / "wake_words.wav"
    custom_wake_word = ""

    directory = wav_path.parent
    os.makedirs(directory, exist_ok=True)
    detector.save_audio(str(wav_path), audio_data)  # 确保传递字符串
    stt = STTMethod()
    text = stt.SenseVoice(str(wav_path))  # 确保传递字符串
    if custom_wake_word in text:
        logging.info(f"text: {text}")
        pass

if __name__ == "__main__":
    # 创建语音唤醒实例
    detector = VoiceWakeWord()
    
    try:
        # 开始监听，设置回调函数
        detector.start_listening(callback=on_wake)
    except Exception as e:
        print(f"发生错误: {e}")
    finally:
        detector.stop_listening()