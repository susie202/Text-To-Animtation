from flask import Flask, request, jsonify

app = Flask(__name__)

# @app.route('/hello_world') #test api
# def hello_world():
#     return 'Hello, World!'

# @app.route('/echo_call/<param>') #get echo api
# def get_echo_call(param):
#     return jsonify({"param": param})

# @app.route('/echo_call', methods=['POST']) #post echo api
# def post_echo_call():
#     param = request.get_json()
#     return jsonify(param)

@app.route('/post_video_url', methods=['POST']) #post echo api
def post_video_url():
    req = request.get_json()
    text = req['action']['detailParams']['text']['value']
    card_id = req['action']['detailParams']['card_id']['value']
    ## Inference_code -> get url
    url = "./test.mp4"
    res = {
        "version": "2.0",
        "template": {
            "outputs": [
                {
                    "simpleText": {
                        "text": url
                        
                    }
                }
            ]
        }
    }
    a = jsonify(res)
    return a

if __name__ == '__main__':
    app.run(host="0.0.0.0", port="5000")