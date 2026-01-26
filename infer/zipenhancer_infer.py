from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

"""
模型地址：https://www.modelscope.cn/models/iic/speech_zipenhancer_ans_multiloss_16k_base/files
"""




# pipeline使用方式

ans = pipeline(
    Tasks.acoustic_noise_suppression,
    model='iic/speech_zipenhancer_ans_multiloss_16k_base')
result = ans(
    'https://modelscope.oss-cn-beijing.aliyuncs.com/test/audios/speech_with_noise1.wav',
    output_path='output.wav')
print("done")




# 流式处理代码示例

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.fileio import File


def create_wav_header(dataflow, sample_rate=16000, num_channels=1, bits_per_sample=16):
    """
    创建WAV文件头的字节串。

    :param dataflow: 音频bytes数据（以字节为单位）。
    :param sample_rate: 采样率，默认16000。
    :param num_channels: 声道数，默认1（单声道）。
    :param bits_per_sample: 每个样本的位数，默认16。
    :return: WAV文件头的字节串和音频bytes数据。
    """
    total_data_len = len(dataflow)
    byte_rate = sample_rate * num_channels * bits_per_sample // 8
    block_align = num_channels * bits_per_sample // 8
    data_chunk_size = total_data_len
    fmt_chunk_size = 16
    riff_chunk_size = 4 + (8 + fmt_chunk_size) + (8 + data_chunk_size)

    # 使用 bytearray 构建字节串
    header = bytearray()

    # RIFF/WAVE header
    header.extend(b'RIFF')
    header.extend(riff_chunk_size.to_bytes(4, byteorder='little'))
    header.extend(b'WAVE')

    # fmt subchunk
    header.extend(b'fmt ')
    header.extend(fmt_chunk_size.to_bytes(4, byteorder='little'))
    header.extend((1).to_bytes(2, byteorder='little'))  # Audio format (1 is PCM)
    header.extend(num_channels.to_bytes(2, byteorder='little'))
    header.extend(sample_rate.to_bytes(4, byteorder='little'))
    header.extend(byte_rate.to_bytes(4, byteorder='little'))
    header.extend(block_align.to_bytes(2, byteorder='little'))
    header.extend(bits_per_sample.to_bytes(2, byteorder='little'))

    # data subchunk
    header.extend(b'data')
    header.extend(data_chunk_size.to_bytes(4, byteorder='little'))

    return bytes(header) + dataflow


ans = pipeline(
    Tasks.acoustic_noise_suppression,
    model='iic/speech_zipenhancer_ans_multiloss_16k_base')

audio_path = 'https://modelscope.oss-cn-beijing.aliyuncs.com/test/audios/speech_with_noise1.wav'

if audio_path.startswith("http"):
    import io

    file_bytes = File.read(audio_path)
    audiostream = io.BytesIO(file_bytes)
else:
    audiostream = open(audio_path, 'rb')

window = 2 * 16000 * 2  # 2 秒的窗口大小，以字节为单位
outputs = b''
total_bytes_len = 0
audiostream.read(44)
for dataflow in iter(lambda: audiostream.read(window), ""):
    print(len(dataflow))
    total_bytes_len += len(dataflow)
    if len(dataflow) == 0:
        break
    result = ans(create_wav_header(dataflow, sample_rate=16000, num_channels=1, bits_per_sample=16))
    output = result['output_pcm']
    outputs = outputs + output
audiostream.close()

outputs = outputs[:total_bytes_len]
output_path = 'output.wav'
with open(output_path, 'wb') as out_wave:
    out_wave.write(create_wav_header(outputs, sample_rate=16000, num_channels=1, bits_per_sample=16))





# 调用提供的onnx模型代码示例

import soundfile as sf
import numpy as np
import torch
import onnxruntime
import io
import os

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.models.audio.ans.zipenhancer import mag_pha_stft, mag_pha_istft
from modelscope.utils.audio.audio_utils import audio_norm
from modelscope.fileio import File
from modelscope.utils.file_utils import get_modelscope_cache_dir


# onnx模型路径
MS_CACHE_HOME = get_modelscope_cache_dir()
onnx_model_path = os.path.join(MS_CACHE_HOME, 'hub/damo/speech_zipenhancer_ans_multiloss_16k_base/onnx_model.onnx')

# 删除旧模型
if os.path.exists(onnx_model_path):
    os.remove(onnx_model_path)

# 下载模型
ans = pipeline(
    Tasks.acoustic_noise_suppression,
    model='iic/speech_zipenhancer_ans_multiloss_16k_base')


audio_path = 'https://modelscope.oss-cn-beijing.aliyuncs.com/test/audios/speech_with_noise1.wav'
output_path = 'output.wav'

is_verbose = True


class OnnxModel:
    def __init__(self, onnx_filepath, providers=None):
        self.onnx_model = onnxruntime.InferenceSession(onnx_filepath, providers=providers)

    def to_numpy(self, tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    def __call__(self, noisy_wav):
        n_fft = 400
        hop_size = 100
        win_size = 400

        norm_factor = torch.sqrt(noisy_wav.shape[1] / torch.sum(noisy_wav ** 2.0))
        if is_verbose:
            print(f"norm_factor {norm_factor}" )
        
        noisy_audio = (noisy_wav * norm_factor)

        noisy_amp, noisy_pha, _ = mag_pha_stft(
            noisy_audio,
            n_fft,
            hop_size,
            win_size,
            compress_factor=0.3,
            center=True)

        ort_inputs = {self.onnx_model.get_inputs()[0].name: self.to_numpy(noisy_amp),
                    self.onnx_model.get_inputs()[1].name: self.to_numpy(noisy_pha),
                    }
        ort_outs = self.onnx_model.run(None, ort_inputs)

        amp_g = torch.from_numpy(ort_outs[0])
        pha_g = torch.from_numpy(ort_outs[1])
        
        if is_verbose:
            print(f"Enhanced amplitude mean and std: {torch.mean(amp_g)} {torch.std(amp_g)}")
            print(f"Enhanced phase mean and std: {torch.mean(pha_g)} {torch.std(pha_g)}")

        wav = mag_pha_istft(
            amp_g,
            pha_g,
            n_fft,
            hop_size,
            win_size,
            compress_factor=0.3,
            center=True)

        wav = wav / norm_factor

        wav = self.to_numpy(wav)

        return wav


onnx_model = OnnxModel(onnx_model_path)

if audio_path.startswith("http"):
    file_bytes = File.read(audio_path)
    wav, fs = sf.read(io.BytesIO(file_bytes))
else:
    wav, fs = sf.read(audio_path)

wav = audio_norm(wav).astype(np.float32)
noisy_wav = torch.from_numpy(np.reshape(wav, [1, wav.shape[0]]))

if is_verbose:
    print(f"wav {wav}")
    print(f"noisy_wav {noisy_wav}")

enhanced_wav = onnx_model(noisy_wav)

if is_verbose:
    print(f"enhanced_wav {enhanced_wav}")
          
sf.write(output_path, (enhanced_wav[0] * 32768).astype(np.int16), fs)