from player.videoPlayer import VideoPlayer

class VideoPlayerWithBubbles(VideoPlayer):
    
    def __init__(self, video_path, subtitle_path = None):
        VideoPlayer.__init__(self, video_path)
        
        self.subtitle_path = subtitle_path