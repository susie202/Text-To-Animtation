from flask import Flask, jsonify

app = Flask(__name__) ## 얘는 python code를 html로 만들어주는 역할을 함!
@app.route('/') ## pwd를 ip:port 로 치환
def test():
    ## 아래처럼 html 문법 사용가능 <a href는 하이퍼링크인데 접속이 안되네요..>
     return '''
    <a href="./test.mp4">Click me.</a> 
    <video autoplay loop>
        <source src="./test.mp4" type="video/mp4">
        <strong>Your browser does not support the video tag.</strong>
    </video>
    </form>
    '''
def test_2():
    


if __name__ == '__main__':
    app.run(host="0.0.0.0", port="5000") ## @app 하위 함수 전체 실행