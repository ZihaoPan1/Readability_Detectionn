import os
import time
import glob
import wave
import torch
import librosa
import numpy as np
import soundfile as sf
import noisereduce as nr
from scipy.io import wavfile
from natsort import natsorted
from moviepy.editor import VideoFileClip, AudioFileClip
from datasets import load_dataset
from speechbrain.pretrained import EncoderClassifier
from transformers import pipeline
from transformers import AutoProcessor, BarkModel
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan


def convert_video2audio(video_file, audio_file='', target_samplerate=16000):
    # video_file = "./Khanmigo.mp4"
    audio_file = audio_file if audio_file else video_file[:-4] + '.mp3'
    target_samplerate = target_samplerate if target_samplerate else 16000
    audio_array = np.array([])
    try:
        video_clip = VideoFileClip(video_file)
        audio_clip = video_clip.audio
        samplerate = audio_clip.fps
        audio_clip.write_audiofile(audio_file, fps=samplerate)
        video_clip.close()
        audio_clip.close()
        if samplerate != target_samplerate:
            audio_array, samplerate = librosa.load(audio_file, sr=samplerate)
            audio_array = librosa.resample(audio_array, orig_sr=samplerate, target_sr=target_samplerate)
            sf.write(audio_file, audio_array, target_samplerate)
    except Exception as e:
        print(e)
    return audio_array


def get_audio_array(video_file, audio_file='', target_samplerate=16000):
    # video_file = "./Khanmigo.mp4"
    os.remove(audio_file)
    return audio_array


def create_speaker_embedding(waveform, speaker_model=None, root_path=''):

    if speaker_model is None:
        spk_model_name = "speechbrain/spkrec-xvect-voxceleb"
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        speaker_model = EncoderClassifier.from_hparams(
            source=spk_model_name,
            run_opts={"device": device},
            savedir=os.path.join(root_path, spk_model_name)
        )

    with torch.no_grad():
        speaker_embeddings = speaker_model.encode_batch(torch.tensor(waveform))
        speaker_embeddings = torch.nn.functional.normalize(speaker_embeddings, dim=2)
        speaker_embeddings = speaker_embeddings.squeeze().cpu().unsqueeze(0)

    return speaker_embeddings, speaker_model





# model_name = 't5'
#
# if model_name == 't5':
#     tts = pipeline(task="text-to-speech", model="microsoft/speecht5_tts", device=0)
#     speaker_embeddings = torch.tensor(embeddings_dataset[2000]["xvector"]).unsqueeze(0)
#     forward_params = {"speaker_embeddings": speaker_embeddings}
# else:
#     tts = pipeline(task="text-to-speech", model="suno/bark-small", device=0)
#     voice_preset = "v2/en_speaker_6"
#     forward_params = {"voice_preset": voice_preset}
#
# start = time.time()
# for i, txt in enumerate(txts):
#     output_path = f"C:/Users/whe/OneDriveMS/wenliang/kids/youtube_videos/test{i + 1}.wav"
#     if model_name == 't5':
#         audio = tts(txt, forward_params=forward_params)
#         sf.write(output_path, audio['audio'], samplerate=audio['sampling_rate'])
#     else:
#         audio = tts(txt)
#         sf.write(output_path, audio['audio'][0], samplerate=audio['sampling_rate'])
# end = time.time()
# dur = end - start
# print(f"Time for completion is {round(dur)} seconds")


#=======================================================================#
class TTS(object):
    def __init__(self,
                 model_name='bark',
                 vocoder=None,
                 speaker_embeddings=None,
                 voice_preset="v2/en_speaker_9"
                 ):
        self.model_name = model_name
        self.vocoder = vocoder
        self.speaker_embeddings = speaker_embeddings
        self.voice_preset = voice_preset

        self.name2model = {
            't5': 'microsoft/speecht5_tts',
            'bark': 'suno/bark-small'
        }
        if model_name not in self.name2model:
            print("Unrecognized model name beyond the scope:", [x for x in self.name2model.values()])
        self.initiate_model()

    def initiate_model(self, model_name='', vocoder=None, speaker_embeddings=None):
        self.model_name = model_name if model_name else self.model_name
        self.vocoder = vocoder if vocoder else self.vocoder
        self.speaker_embeddings = speaker_embeddings if speaker_embeddings else self.speaker_embeddings
        self.processor = AutoProcessor.from_pretrained(self.name2model[self.model_name])

        if self.model_name.startswith('t5'):
            if self.vocoder is None:
                self.vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

            if self.speaker_embeddings is None:
                embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
                self.speaker_embeddings = torch.tensor(embeddings_dataset[2000]["xvector"]).unsqueeze(0)

            self.model = SpeechT5ForTextToSpeech.from_pretrained(self.name2model[self.model_name])
            self.samplerate = 16000

        elif self.model_name.startswith('bark'):
            self.model = BarkModel.from_pretrained(self.name2model[self.model_name])
            self.samplerate = self.model.generation_config.sample_rate

        else:
            print("Unrecognized model name: Please check allowed model names with self.name2model")

        print("model loaded:", self.name2model[self.model_name])

        if torch.cuda.is_available():
            self.model.to("cuda:0")
            self.device = "cuda:0"
        else:
            self.device = "cpu"

        print("model loaded into", self.device)

    def text2speech(self, text):

        if self.model_name.startswith('t5'):
            inputs = self.processor(text=text, return_tensors="pt")
            audio = self.model.generate_speech(
                inputs['input_ids'].to(self.device),
                self.speaker_embeddings.to(self.device),
                vocoder=self.vocoder.to(self.device)
            )
        elif self.model_name.startswith('bark'):
            inputs = self.processor(text=text, voice_preset=self.voice_preset)
            audio = self.model.generate(**inputs.to(self.device))

        audio = audio.cpu().numpy().squeeze()

        return audio

    def save_wav(self, audio, samplerate, output_path):
        assert output_path.endswith('.wav'), "file must be in the .wav format"
        sf.write(output_path, audio, samplerate=samplerate)

    def convert_txts(self, txts):
        start = time.time()
        for i, txt in enumerate(txts):
            audio = self.text2speech(txt)
            output_path = f"./temp_sst{i + 1}.wav"
            self.save_wav(audio, self.samplerate, output_path)
            print(f"file saved: {output_path}")
        end = time.time()
        dur = end - start
        print(f"wav files produced in {round(dur)} seconds")

    def concatenate_wavs(self, wav_files, output_filename=''):
        """ Concatenates wav files into one wav file """

        # Open the first input file to get parameters
        with wave.open(wav_files[0], 'rb') as input_wav:
            # input_wav = wave.open(wav_files[0], 'rb')
            params = input_wav.getparams()
            nframes = sum([wave.open(f, 'rb').getnframes() for f in wav_files])

        # Open the output file
        with wave.open(output_filename, 'wb') as output_wav:
            # Set the parameters for the output file
            output_wav.setparams(params)
            output_wav.setnframes(nframes)
            # Write frames from each input file to the output file
            for file in wav_files:
                with wave.open(file, 'rb') as input_wav:
                    data = input_wav.readframes(input_wav.getnframes())
                    output_wav.writeframes(data)


run = False
create_speaker_embeddings = False

if run:
    root_path = "C:/Users/whe/OneDriveMS/wenliang/kids/youtube_videos"
    os.chdir(root_path)

    if create_speaker_embeddings:
        audio_array = convert_video2audio(video_file="Khanmigo.mp4")
        audio_array = convert_video2audio(video_file="Khanmigo.mp4", audio_file="Khanmigo.wav")
        audio_array = convert_video2audio(video_file="Khanmigo.mp4", audio_file="Khanmigo.wav", target_samplerate=44100)

        speaker_embeddings, speaker_model = create_speaker_embedding(audio_array)
        self = TTS(model_name='t5', speaker_embeddings=speaker_embeddings)
    else:
        self = TTS(model_name='t5')

    # convert texts to wav files
    text1 = "These are just a few examples, and I am sure you can imagine many more!"
    text2 = "However, with so much power comes the responsibility, and it is important to highlight that TTS models have the potential to be used for malicious purposes."
    text3 = "For example, with sufficient voice samples, malicious actors could potentially create convincing fake audio recordings, leading to the unauthorized use of someone’s voice for fraudulent purposes or manipulation."
    text4 = "If you plan to collect data for fine-tuning your own systems, carefully consider privacy and informed consent."
    txts = [text1, text2, text3, text4]
    self.convert_txts(txts)

    # combine wav files
    filename = "t5_test_again.wav"
    wav_files = natsorted(glob.glob(f"./temp_sst*.wav"))
    self.concatenate_wavs(wav_files, output_filename=f"./{filename}")
    [os.remove(x) for x in wav_files]

    # reduce background noise
    # https://msstash.morningstar.com/users/bsheraf/repos/transcription-evaluation/browse/audio_enhancement.ipynb
    filename = "Khanmigo.wav"
    rate, audio_data = wavfile.read(f"./{filename}")
    print(rate)
    audio, rate = librosa.load(f"./{filename}", sr=None)
    print(rate)
    nbits = 16
    audio_data = audio_data / 2 ** (nbits - 1)
    print(all(audio_data == audio))
    reduced_audio = nr.reduce_noise(y=audio_data, sr=rate)
    wavfile.write(f"./{filename[:-4]}" + '_reduced.wav', rate, reduced_audio)

    # ii = [7306] + [x*1000 for x in range(len(embeddings_dataset) // 1000)]
    # for i in ii:
    #     speaker_embeddings = torch.tensor(embeddings_dataset[i]["xvector"]).unsqueeze(0)
    #     speech = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)
    #     output_path = f"./test_speaker_embeddings{i}.wav"
    #     sf.write(output_path, speech.numpy(), samplerate=16000)


