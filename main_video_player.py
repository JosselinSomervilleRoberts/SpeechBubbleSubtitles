#from player.videoPlayerWithBubbles import VideoPlayerWithBubbles as Player
from player.videoPlayerMesh import VideoPlayerMesh as Player

if __name__ == '__main__':
    vid_play = Player("data/video.mp4")#, "data/video.ass", "data/faces")
    vid_play.play()
