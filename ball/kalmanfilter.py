import numpy as np
from filterpy.kalman import KalmanFilter

class KalmanBoxTracker(object):
  """
  卡尔曼滤波器
  """
  def __init__(self,bbox):
    """
    用[x,y,s,r]初始化
    """
    #define constant velocity model
    self.kf = KalmanFilter(dim_x=7, dim_z=4) 
    self.kf.F = np.array([[1,0,0,0,1,0,0],[0,1,0,0,0,1,0],[0,0,1,0,0,0,1],[0,0,0,1,0,0,0],  [0,0,0,0,1,0,0],[0,0,0,0,0,1,0],[0,0,0,0,0,0,1]])
    self.kf.H = np.array([[1,0,0,0,0,0,0],[0,1,0,0,0,0,0],[0,0,1,0,0,0,0],[0,0,0,1,0,0,0]])

    self.kf.R[2:,2:] *= 10.
    self.kf.P[4:,4:] *= 1000. #give high uncertainty to the unobservable initial velocities
    self.kf.P *= 10.
    self.kf.Q[-1,-1] *= 0.01
    self.kf.Q[4:,4:] *= 0.01

    self.kf.x[:4] = bbox.reshape(4,1)
    self.time_since_update = 0
    self.history = []
    self.hits = 0
    self.hit_streak = 0
    self.age = 0

  def update(self,bbox):
    """
    传入参数bbox为新检测到的信息[x,y,s,r]
    """
    self.time_since_update = 0
    self.history = []
    self.hits += 1
    self.hit_streak += 1
    self.kf.update(bbox.reshape(4,1))

  def predict(self):
    """
    预测下一帧的信息[x,y,s,r],注意返回的时候为转置向量，前四个分别对应x,y,s,r
    """
    if((self.kf.x[6]+self.kf.x[2])<=0):
      self.kf.x[6] *= 0.0
    self.kf.predict()
    self.age += 1
    if(self.time_since_update>0):
      self.hit_streak = 0
    self.time_since_update += 1
    self.history.append(self.kf.x)
    return self.history[-1]

  def get_state(self):
    """
    在用当前时刻的检测值更新之后输出融合预测值与检测值的最优估计
    """
    return self.kf.x


# kalman = KalmanBoxTracker(np.array([1,1,1,1]))
# print(kalman.predict())
# kalman.update(np.array([1.,1.1,1,1]))
# print(kalman.predict())
# kalman.update(np.array([1.1,1.5,1.1,1]))
# print(kalman.predict())