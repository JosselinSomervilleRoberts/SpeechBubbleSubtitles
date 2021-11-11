from player.recognizer.interpolable import Interpolable



class Position:
    
    def __init__(self):
        self.x = Interpolable()
        self.y = Interpolable()
        
        
    def add(self, value, frame_index):
        self.x.add(value[0], frame_index)
        self.y.add(value[1], frame_index)
        
        
    def get(self, frame_index):
        return (self.x.get(frame_index), self.y.get(frame_index))
    
    
    def cleanup(self, frame_index):
        self.x.cleanup(frame_index)
        self.y.cleanup(frame_index)

    
    def merge(self, other):
        self.x.merge(other.x)
        self.y.merge(other.y)