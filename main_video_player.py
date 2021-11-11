# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 16:05:06 2021

@author: josse
"""

from player.videoPlayerWithBubbles import VideoPlayerWithBubbles as Player

vid_play = Player("data/video.mp4", "data/video.ass")
vid_play.play()