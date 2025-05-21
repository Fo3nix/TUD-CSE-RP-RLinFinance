# RQ3/rewards/global_history.py

class GlobalHistory:
    def __init__(self, max_len=10000):
        self.data = []
        self.max_len = max_len

    def append(self, pnl):
        self.data.append(pnl)
        if len(self.data) > self.max_len:
            self.data.pop(0)

    def get(self, window):
        if len(self.data) < window:
            return self.data
        return self.data[-window:]

    def reset(self):
        self.data = []

GLOBAL_HISTORY = GlobalHistory(max_len=10000)
