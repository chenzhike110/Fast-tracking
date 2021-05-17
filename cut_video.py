from moviepy.editor import *

video = "../SRTP/video/DJI_0256.MP4"
clip1 = VideoFileClip(video).subclip(38,48)
clip1.write_videofile("offside1.mp4")