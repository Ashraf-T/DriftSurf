"""
The Tornado Framework
By Ali Pesaranghader
University of Ottawa, Ontario, Canada
E-mail: apesaran -at- uottawa -dot- ca / alipsgh -at- gmail -dot- com
"""

import time
#import models

class SuperDetector:
    """A drift detector method inherits this super detector class!"""

    def __init__(self):
        self.RUNTIME = 0
        self.TOTAL_RUNTIME = 0

    def detect(self, pr):
        t1 = time.perf_counter()
        warning_status, drift_status = self.run(pr)
        t2 = time.perf_counter()
        delta_t = (t2 - t1) * 1000  # in milliseconds
        self.RUNTIME += delta_t
        self.TOTAL_RUNTIME += delta_t
        return warning_status, drift_status

    def run(self, pr):
        return False, False

    def reset(self):
        self.RUNTIME = 0

    def get_settings(self):
        raise NotImplementedError('THE RESET FUNCTION HAS NOT BEEN DEFINED IN THE CHILD')

    def test(self, model, test_set):
        for (i, x, y) in test_set:
            y_hat = model.predict(x)
            _, drift_status = self.detect(y == y_hat)
            if drift_status:
                return True
        return False
