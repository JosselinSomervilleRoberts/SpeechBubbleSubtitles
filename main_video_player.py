from player.videoPlayerWithBubbles import VideoPlayerWithBubbles as Player

if __name__ == '__main__':
    vid_play = Player("data/video.mp4", "data/video.ass")
    vid_play.play()