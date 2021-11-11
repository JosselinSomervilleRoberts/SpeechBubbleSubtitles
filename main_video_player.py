from multiprocessing.pool import ThreadPool

from player.videoPlayerWithBubbles import VideoPlayerWithBubbles as Player

if __name__ == '__main__':
    pool = ThreadPool(processes = 2)
    vid_play = Player("data/video.mp4", "data/video.ass", pool)
    vid_play.play()