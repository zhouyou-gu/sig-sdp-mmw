import math

import numpy as np


class RangeBinarySearch:
    def __init__(self, start, end):
        assert start is int
        assert end is int
        self.start = start  # Start of the range (inclusive)
        self.end = end      # End of the range (inclusive)
        self.it = 0

        self.feasibility_check_alg = None
    def search(self):
        self.it = 0
        left, right = self.start, self.end

        while True:
            self.it += 1
            mid = math.floor(float(left+right)/2.)
            ret = self.call_feasibility_check()
            if ret:
                right = mid
            else:
                left = mid+1
            if left >= right:
                break
        return right

    def call_feasibility_check(self) -> bool:
        pass