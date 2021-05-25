import cv2, queue, threading, time

# bufferless VideoCapture
class VideoCapture:

  def __init__(self, name):
    self.cap = cv2.VideoCapture(name)
    self.q = queue.Queue()
    t = threading.Thread(target=self._reader)
    t.daemon = True
    t.start()

  # read frames as soon as they are available, keeping only most recent one
  def _reader(self):
    while True:
      ret, frame = self.cap.read()
      if not ret:
        break
      if not self.q.empty():
        try:
          self.q.get_nowait()   # discard previous (unprocessed) frame
        except queue.Empty:
          pass
      self.q.put(frame)

  def read(self):
    return self.q.get()
  
  def get(self):
    return self.cap.get(cv2.CAP_PROP_FPS)
  
  def release(self):
    self.cap.release()


if __name__ == '__main__':
  # cap = cv2.VideoCapture("rtmp://127.0.0.1:9999/live/test")
  cap2=VideoCapture("rtmp://127.0.0.1:9999/live/test")
  while True:
    # time.sleep()   # simulate time between events
    frame1 = cap2.read()
    print(frame1.shape)
    # _,frame = cap.read()
    # cv2.imshow("frame", frame)
    cv2.imshow("frame2",frame1)
    if chr(cv2.waitKey(1)&255) == 'q':
      break
