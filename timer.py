import time

class timer:
    def __init__ (self):
        self.reset()
    
    def start (self):
        if self.running:
            return
        self.running = True
        self.start_time = time.time()
    
    def stop (self):
        if not self.running:
            return
        self.running = False
        self.last_measured_time = (time.time()-self.start_time) + self.last_measured_time
    
    def get_time (self):
        if self.running:
            return (time.time()-self.start_time) + self.last_measured_time
        else:
            return self.last_measured_time
    
    def reset (self):
        self.last_measured_time = 0
        self.start_time = None
        self.running = False
