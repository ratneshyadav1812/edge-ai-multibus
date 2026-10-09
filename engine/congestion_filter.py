import math

class EdgeCongestionFilter:
    def __init__(self, sustain_time=5, cooldown_time=15, dist_threshold=500):
        self.sustain_time = sustain_time       
        self.cooldown_time = cooldown_time     
        self.dist_threshold = dist_threshold   

        self.congestion_start_time = None
        self.last_sent_time = 0
        self.last_sent_x = 0.0
        self.last_sent_y = 0.0

    def calculate_distance(self, x1, y1, x2, y2):
        return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
