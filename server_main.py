import os
import threading
import soundfile
from flask import Flask, jsonify, request, render_template
from urllib import parse as url_parse
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

class Inference(threading.Thread):
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
    while (True):
      if (len(inference_queue) > 0):
        text, card_id = inference_queue[0]
        wav_path = data_dir + "/voice_data/{}_{}.wav".format(text, str(card_id)).replace(" ","_")
        video_path = data_dir + "/video_data/{}_{}.mp4".format(text, str(card_id))
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
        # inference_processing = False


class Communication(threading.Thread):
  app = Flask(__name__, instance_path="/home/whjung/Com2us/IntegratedCode")
  res = ""
  host = ""
  port = ""

  def __init__(self, host="0.0.0.0", port="5000"):
    super().__init__()
    self.host = host
    self.port = port

  @app.route('/index', methods=['POST'])
  def post_video_url():
    global inference_queue
    req = request.get_json()
    text = req['action']['detailParams']['text']['value']
    encoded_text = url_parse.quote(text)
    card_id = int(req['action']['detailParams']['card_id']['value'])
    queue_length = len(inference_queue)
    url = "http://121.144.73.146:50000/wait_second/{}_{}".format(encoded_text, str(card_id))
    description = "입력하신 텍스트에 따라 합성이 오래걸릴 수 있습니다.\n현재 {}명이 대기중입니다.\n예상 대기시간은 {}초 입니다.".format(queue_length, (
            queue_length + 1) * 7)
    res = {
      "version": "2.0",
      "template": {
        "outputs": [
          {
            "carousel": {
              "type": "basicCard",
              "items": [
                {
                  "title": "표정합성 결과 입니다!",
                  "description": description,
                  "thumbnail": {
                    "imageUrl": "https://play-lh.googleusercontent.com/le3cTY5i5BOliCsSwZPw0KcBJdB4r3t9t7G-QXXW6YkbXj75dxHlTiFrWm7uK-xD7Q=s180-rw"
                  },
                  "buttons": [
                    {
                      "action": "webLink",
                      "label": "동영상 보기",
                      "webLinkUrl": url
                    }
                  ]
                }
              ]
            }
          }
        ]
      }
    }
    video_path = data_dir + "/video_data/{}_{}.mp4".format(text, str(card_id))
    if not os.path.exists(video_path):
      inference_queue.append([text, card_id])
    return jsonify(res)

  @app.route("/get_len/<encoded_text>_<card_id>", methods=['GET'])
  def get_len(encoded_text, card_id):
    global inference_queue
    queue_length = len(inference_queue)
    ## 여기를 바꿔야겠네요.
    text = url_parse.unquote(encoded_text)
    order_index = 0
    while (True):  ## 위험..한게.. index++ 되었는데 pop 해버려서, index이전으로 encoded_text가 보인다면...
      if (inference_queue[order_index] == [text, int(card_id)]):
        break
      order_index += 1
      if (order_index > queue_length):  ## index이전으로 encoded_text가 보인다면 -> 다시 돌려서 언젠가는 찾게만들기!
        order_index = 0
    return str(order_index)

  @app.route("/wait_second/<encoded_text>_<card_id>", methods=['GET'])
  def wait_html(encoded_text, card_id):
    encoded_text = url_parse.quote(encoded_text)
    return render_template('wait_design.html', encoded_text=encoded_text, card_id=card_id)

  @app.route("/test_video/<encoded_text>_<card_id>", methods=['GET'])
  def pass_html(encoded_text, card_id):
    text = url_parse.unquote(encoded_text)
    video_path = data_dir + "/video_data/{}_{}.mp4".format(text, str(card_id))
    while (True):
      if os.path.exists(video_path):
        break
    return render_template('test_video.html', video_name=os.path.basename(video_path),
                           card_path=str(card_id) + ".png")

  def run(self):
    self.app.run(host=self.host, port=self.port, threaded='True')


if __name__ == '__main__':
  print("Server Start!\n")
  thread_infer = Inference()
  thread_communication = Communication(host="0.0.0.0", port="5000")

  thread_infer.start()
  thread_communication.start()

  thread_infer.join()
  thread_communication.join()

  print("Not Reached Here!\n")

############################################################## Server ##############################################################


