# IntegratedCode
통합코드!

#사용방법
server_main.py 실행이후 -> 카카오톡 봇으로 실행


# 사용시 바꿀 사항
<ol>

####server_main.py
<ol>
1. 13번째 줄 : data_dir<br>
2. 65번째 줄 : instance_path = IntegratedCode 위치<br>
3. 83번째 줄 : url = http://121.144.73.146:50000 -> http://[flask_server_ip]<br>
    <ol>*이건 kakaotalk 봇서버에도 바꿔야해서 만약 그냥 그대로 테스트 해보실거면
        @정우현 에게 문의주십쇼~~ 바로 켜드릴게요.
        <br> 
        참고로 저는 121.144.73.146은 공인 ip이고 공유기에서 포트 50000 을 통해서 와이파이 연결된 제 Local ip : 50000 로 포트 포워딩 했구요,
        nginx에서 50000번을 다시 5000으로 포워딩해줘서 사용중입니다.
    </ol>
</ol>

####wait_design.html
<ol>
1. 87번째 줄 : url에 위와같이 마찬가지로 개인 서버ip로 변경
</ol>
</ol>

<br>

### 주의사항
<ol>
    1. server_main.py는 봇을 만들어야 합니다.<br>
    2. client_main.py는 바로 실행되나, pyqt로 띄우는건 안해서 경로만 보여줍니다.<br>
    3. 방화벽 문제
    <ol>
    방화벽 정지 : systemctl stop firewalld
    </ol>
</ol>

