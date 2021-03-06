import cv2
from flask import Flask, render_template, Response
from camera import VideoCamera

app = Flask(__name__)

video_stream = VideoCamera()

@app.route('/')
def index():
    return render_template('index.html')

def gen(camera):
    while True:
        ret, frame = camera.get_frame()
        if not ret:
            break
        else:
            yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
        return Response(gen(video_stream),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='127.0.0.1', debug=True,port="5000")