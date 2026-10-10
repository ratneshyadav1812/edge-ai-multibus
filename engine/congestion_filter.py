import math

class EdgeCongestionFilter:
    def __init__(self, sustain_time=5, cooldown_time=15, dist_threshold=500):
        # We are using shorter times for testing purposes:
        # sustain_time: must see high traffic for 5 seconds to confirm
        # cooldown_time: wait 15 seconds before sending another alert
        self.sustain_time = sustain_time       
        self.cooldown_time = cooldown_time     
        self.dist_threshold = dist_threshold   

        self.congestion_start_time = None
        self.last_sent_time = 0
        self.last_sent_x = 0.0
        self.last_sent_y = 0.0

    def calculate_distance(self, x1, y1, x2, y2):
        return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    def process_frame(self, current_status, current_time, current_x, current_y):
        # 1. Reset if traffic drops
        if current_status != "High":
            self.congestion_start_time = None
            return False

        # 2. Start tracking sustained congestion
        if self.congestion_start_time is None:
            self.congestion_start_time = current_time
            return False 

        time_congested = current_time - self.congestion_start_time

        # 3. Check Debounce, Cooldown, and Distance
        if time_congested >= self.sustain_time:
            time_since_last_send = current_time - self.last_sent_time
            distance_from_last_send = self.calculate_distance(self.last_sent_x, self.last_sent_y, current_x, current_y)

            if time_since_last_send >= self.cooldown_time or distance_from_last_send >= self.dist_threshold:
                # Trigger the event and reset timers
                self.last_sent_time = current_time
                self.last_sent_x = current_x
                self.last_sent_y = current_y
                self.congestion_start_time = None 
                
                return True 

        return False