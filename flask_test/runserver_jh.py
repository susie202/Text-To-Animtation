from flask import Flask, jsonify, request, Response
import os, re

app = Flask(__name__)  ## 얘는 python code를 html로 만들어주는 역할을 함!


# 183.38.1.167:5001/video


@app.route("/coffee", methods=["POST"])
def coffee():
    req = request.get_json()
    coffee_menu = req["action"]["detailParams"]["coffee_menu"]["value"]
    answer = coffee_menu

    res = {
        "version": "2.0",
        "template": {
            "outputs": [
                {
                    "listCard": {
                        "header": {"title": "카카오 i 디벨로퍼스를 소개합니다"},
                        "items": [
                            {
                                "title": "Kakao i Developers",
                                "description": "새로운 AI의 내일과 일상의 변화",
                                "imageUrl": "http://k.kakaocdn.net/dn/APR96/btqqH7zLanY/kD5mIPX7TdD2NAxgP29cC0/1x1.jpg",
                                "link": {
                                    "web": "https://namu.wiki/w/%EB%9D%BC%EC%9D%B4%EC%96%B8(%EC%B9%B4%EC%B9%B4%EC%98%A4%ED%94%84%EB%A0%8C%EC%A6%88)"
                                },
                            },
                            {
                                "title": "Kakao i Open Builder",
                                "description": "카카오톡 채널 챗봇 만들기",
                                "imageUrl": "http://k.kakaocdn.net/dn/N4Epz/btqqHCfF5II/a3kMRckYml1NLPEo7nqTmK/1x1.jpg",
                                "link": {
                                    "web": "https://namu.wiki/w/%EB%AC%B4%EC%A7%80(%EC%B9%B4%EC%B9%B4%EC%98%A4%ED%94%84%EB%A0%8C%EC%A6%88)"
                                },
                            },
                            {
                                "title": "Kakao i Voice Service",
                                "description": "보이스봇 / KVS 제휴 신청하기",
                                "imageUrl": "http://k.kakaocdn.net/dn/bE8AKO/btqqFHI6vDQ/mWZGNbLIOlTv3oVF1gzXKK/1x1.jpg",
                                "link": {
                                    "web": "https://namu.wiki/w/%EC%96%B4%ED%94%BC%EC%B9%98"
                                },
                            },
                        ],
                        "buttons": [
                            {
                                "label": "구경가기",
                                "action": "webLink",
                                "webLinkUrl": "http://118.38.1.167:5001/video",
                            }
                        ],
                    }
                }
            ]
        },
    }

    return jsonify(res)


"""
    res = {
        "version": "2,0",
        "template": {
            "outputs": [
                {
                    "liskCard": {
                        "header": {"title": "배고파요"},
                        "items": [
                            {
                                "title": "kako",
                                "description": "배고푸ㅡ다니까",
                                "link": {"https://namu.wiki/w/%EB%9D%BC%EC%9D%B4%EC%96%B8(%EC%B9%B4%EC%B9%B4%EC%98%A4%ED%94%84%EB%A0%8C%EC%A6%88)"},
                            }
                        ],
                        "buttons": [
                            {
                                "label": "밥줘",
                                "action": "webLink",
                                "webLinkUrl": "https://namu.wiki/w/%EB%9D%BC%EC%9D%B4%EC%96%B8(%EC%B9%B4%EC%B9%B4%EC%98%A4%ED%94%84%EB%A0%8C%EC%A6%88)",
                            }
                        ],
                    }
                }
            ]
        },
    }
    """


#    return jsonify(res)


@app.after_request
def after_request(response):
    response.headers.add("Accept-Ranges", "bytes")
    return response


def get_chunk(byte1=None, byte2=None):
    full_path = "test.mp4"
    file_size = os.stat(full_path).st_size
    start = 0

    if byte1 < file_size:
        start = byte1
    if byte2:
        length = byte2 + 1 - byte1
    else:
        length = file_size - start

    with open(full_path, "rb") as f:
        f.seek(start)
        chunk = f.read(length)
    return chunk, start, length, file_size


@app.route("/video")
def get_file():
    range_header = request.headers.get("Range", None)
    byte1, byte2 = 0, None
    if range_header:
        match = re.search(r"(\d+)-(\d*)", range_header)
        groups = match.groups()

        if groups[0]:
            byte1 = int(groups[0])
        if groups[1]:
            byte2 = int(groups[1])

    chunk, start, length, file_size = get_chunk(byte1, byte2)
    resp = Response(
        chunk,
        206,
        mimetype="video/mp4",
        content_type="video/mp4",
        direct_passthrough=True,
    )

    resp.headers.add(
        "Content-Range",
        "bytes {0}-{1}/{2}".format(start, start + length - 1, file_size),
    )
    return resp


if __name__ == "__main__":
    app.run(host="0.0.0.0", port="5001", threaded=True)  ## @app 하위 함수 전체 실행
