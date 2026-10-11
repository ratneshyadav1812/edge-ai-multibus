import unittest
from engine.congestion_filter import EdgeCongestionFilter

class TestEdgeCongestionFilter(unittest.TestCase):
    def setUp(self):
        self.filter = EdgeCongestionFilter(sustain_time=5, cooldown_time=15, dist_threshold=500)

    def test_low_traffic_returns_false(self):
        self.assertFalse(self.filter.process_frame("Low", current_time=1.0, current_x=0.0, current_y=0.0))

    def test_sustained_traffic_and_cooldown(self):
        # Initial high frame starts tracking
        self.assertFalse(self.filter.process_frame("High", current_time=20.0, current_x=0.0, current_y=0.0))
        # 3 seconds later: not yet sustained
        self.assertFalse(self.filter.process_frame("High", current_time=23.0, current_x=0.0, current_y=0.0))
        # 5 seconds later: sustained threshold reached -> triggers event
        self.assertTrue(self.filter.process_frame("High", current_time=25.0, current_x=0.0, current_y=0.0))

        # Immediately afterwards: cooldown prevents trigger
        self.assertFalse(self.filter.process_frame("High", current_time=26.0, current_x=0.0, current_y=0.0))
        # Cooldown still active at t=31 (elapsed 6s < 15s)
        self.assertFalse(self.filter.process_frame("High", current_time=31.0, current_x=0.0, current_y=0.0))
        
        # After cooldown elapsed (at t=40, 15s elapsed >= 15s cooldown) -> triggers
        self.assertTrue(self.filter.process_frame("High", current_time=40.0, current_x=0.0, current_y=0.0))

    def test_distance_based_trigger_override(self):
        # Trigger first event
        self.filter.process_frame("High", current_time=20.0, current_x=0.0, current_y=0.0)
        self.assertTrue(self.filter.process_frame("High", current_time=25.0, current_x=0.0, current_y=0.0))

        # Within cooldown, but moved far (> 500 units)
        self.filter.process_frame("High", current_time=26.0, current_x=600.0, current_y=600.0)
        self.assertTrue(self.filter.process_frame("High", current_time=31.0, current_x=600.0, current_y=600.0))

if __name__ == "__main__":
    unittest.main()
