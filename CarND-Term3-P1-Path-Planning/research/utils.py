import numpy as np
import pandas as pd
import math


def distance(x1, y1, x2, y2):
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

class Highway(object):
    def __init__(self, highway_csv):
        self._highway_df = pd.read_csv(highway_csv,
                                       delimiter = ' ',
                                       names = ['x', 'y', 's', 'dx', 'dy'])

    def closestWaypoint(self, x, y):
        return np.argmin(distance(x, y,
                                  self._highway_df.x, self._highway_df.y))

    def closestWaypointXY(self, x, y):
        iClosest = self.closestWaypoint(x, y)
        return self._highway_df.x[iClosest], self._highway_df.y[iClosest]

    def nextWaypoint(self, x, y, theta):
        iClosest = self.closestWaypoint(x, y)

        map_x = self._highway_df.x[iClosest]
        map_y = self._highway_df.y[iClosest]

        heading = math.atan2(map_y - y, map_x - x)
        angle = math.fabs(heading - theta)
        angle = min(2.*math.pi - angle, angle)

        if (angle > math.pi/4):
            iClosest += 1

            if (iClosest >= self._highway_df.shape[0]):
                iClosest = 0
        return iClosest

    def getFrenet(self, x, y, theta):
        inext = self.nextWaypoint(x, y, theta)
        iprev = inext - 1

        n_x = self._highway_df.x[inext] - self._highway_df.x[iprev]
        n_y = self._highway_df.y[inext] - self._highway_df.y[iprev]

        x_x = x - self._highway_df.x[iprev]
        x_y = y - self._highway_df.y[iprev]

        # project of x on n
        proj_norm = (x_x * n_x + x_y * n_y) / (n_x**2 + n_y**2)
        proj_x = proj_norm * n_x
        proj_y = proj_norm * n_y

        # Frenet d
        frenet_d = distance(x_x, x_y, proj_x, proj_y)

        # to see if d is positive or negative, we compare it to a center point
        center_x = 1000 - self._highway_df.x[iprev]
        center_y = 2000 - self._highway_df.y[iprev]

        center_to_pos = distance(center_x, center_y, x_x, x_y)
        center_to_ref = distance(center_x, center_y, proj_x, proj_y)

        if (center_to_pos <= center_to_ref):
            frenet_d = -frenet_d

        # Frenet s = distance of path from 0 to iprev point + last project distance
        diff_dist = np.sqrt(  np.diff(self._highway_df.x[:iprev])**2
                            + np.diff(self._highway_df.y[:iprev])**2)

        frenet_s = np.sum(diff_dist) + distance(0, 0, proj_x, proj_y)

        return frenet_s, frenet_d

    def getXY(self, s, d):
        if (s >= self._highway_df.s.values[-1]):
            prev_wp = len(self._highway_df.s) - 1
        else:
            prev_wp = np.argmax(self._highway_df.s > s) - 1

        wp = (prev_wp + 1) % len(self._highway_df.s)

        heading = math.atan2(self._highway_df.y[wp] - self._highway_df.y[prev_wp],
                             self._highway_df.x[wp] - self._highway_df.x[prev_wp])

        seg_s = s - self._highway_df.s[prev_wp]
        seg_x = self._highway_df.x[prev_wp] + seg_s * math.cos(heading)
        seg_y = self._highway_df.y[prev_wp] + seg_s * math.sin(heading)

        perp_heading = heading - math.pi/2.
        x = seg_x + d * math.cos(perp_heading)
        y = seg_y + d * math.sin(perp_heading)

        return x, y