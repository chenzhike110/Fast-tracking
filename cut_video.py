from moviepy.editor import *

video = "../SRTP/video/DJI_0259.MP4"
clip1 = VideoFileClip(video).subclip(135,160)
clip1.write_videofile("noneoffside259.mp4")
