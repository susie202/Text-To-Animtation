import os
import soundfile
import kobert_emo.run_kobert as kobert
import tts
import face2video

############################################################## Server ##############################################################
## parameter


data_dir = '/home/whjung/Com2us/IntegratedCode/data'

## global variable
# inference_processing = False
inference_queue = []
##

class Inference():
  def __init__(self):
    super().__init__()
    ## TODO : model load
    self.kobert_model = kobert.model_load('./models/kobert_emo')
    self.tacotron_model_mlb, self.waveglow_model = tts.model_load(
      tacotron2_path='./models/tts/tacotron2_model/checkpoint_155000_mlb',waveglow_path='./models/tts/waveglow_model/waveglow_54000')
    self.tacotron_model_hap = tts.tacotron2_load(
      tacotron2_path='./models/tts/tacotron2_model/hap/checkpoint_20000_hap')
    self.tacotron_model_sad = tts.tacotron2_load(
      tacotron2_path='./models/tts/tacotron2_model/sad/checkpoint_40000')
    self.tacotron_model_ang = tts.tacotron2_load(
      tacotron2_path='./models/tts/tacotron2_model/ang/process/checkpoint_25000')
    self.face2video_model, self.face2video_opt= face2video.model_load('./models/face2video')

  def run(self):
    emotion_list = ['hap','sad','ang']
    # global inference_processing
    global inference_queue
    if (len(inference_queue) > 0):
      text, card_id = inference_queue[0]
      wav_path = data_dir + "/voice_data/{}_{}.wav".format(text, str(card_id)).replace(" ","_")
      video_path = data_dir + "/video_data/{}_{}.mp4".format(text, str(card_id)).replace(" ","_")
      if card_id != 0:
        emotions = kobert.predict(input=text , model=self.kobert_model)
        emotion = emotion_list[emotions[0]]
        image_path = data_dir + '/card_data/' + str(card_id) + os.sep + emotion
        if emotion == 'hap':
          tacotron2_model = self.tacotron_model_hap
        elif emotion == 'sad':
          tacotron2_model = self.tacotron_model_sad
        else:
          tacotron2_model = self.tacotron_model_ang
      else:
        tacotron2_model = self.tacotron_model_mlb
        image_path = data_dir + '/card_data/'+ str(card_id)
      wav = tts.inference(text.replace("."," ")+".", tacotron2_model,self.waveglow_model)
      soundfile.write(wav_path,wav,samplerate=22050, format='WAV', endian='LITTLE')
      _ = face2video.inference(wav_path,image_path,self.face2video_model,self.face2video_opt,video_path)
      _ = inference_queue.pop(0)  ## 이유 - 한명이 실행중일때, 그다음사람이 바로 0명대기중이라고 뜨기 때문
      return video_path

if __name__ == '__main__':
  print("Client Start!\n")
  video_path = ""
  infer = Inference()
  while(True):
    os.system('clear')
    if (len(video_path)>0):
      print("Saved as \'"+video_path+"\'")
    text = input("텍스트를 입력하세요 : ")
    card_id = int(input("선수 카드를 입력하세요(0, 10172, 100451, ...) : "))
    inference_queue.append([text,card_id])
    video_path = infer.run()
  print("Client End!\n")

############################################################## Server ##############################################################


