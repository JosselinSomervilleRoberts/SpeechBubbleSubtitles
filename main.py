from player import VideoPlayerWithBubbles as Player
#from player import VideoPlayerMesh as Player
#from player import VideoPlayer as Player

if __name__ == '__main__':
    vid_play = Player("data/video.mp4", "data/video.ass", "data/faces")
    vid_play.play()

