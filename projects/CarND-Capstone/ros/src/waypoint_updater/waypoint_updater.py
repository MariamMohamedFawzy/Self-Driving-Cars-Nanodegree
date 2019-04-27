#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped, TwistStamped
from styx_msgs.msg import Lane, Waypoint
from std_msgs.msg import Int32
from scipy.spatial import KDTree
import numpy as np

import math

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.

TODO (for Yousuf and Aaron): Stopline location for each traffic light.
'''

LOOKAHEAD_WPS = 20 # Number of waypoints we will publish. You can change this number
MAX_DECEL = .5



class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater')

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)
        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb)
        # rospy.Subscriber('/current_velocity', TwistStamped, self.current_velocity_cb)
        # rospy.Subscriber('/obstacle_waypoint', Lane, self.obstacle_waypoint_cb)

        # TODO: Add a subscriber for /traffic_waypoint and /obstacle_waypoint below
        # MAX_DECEL = rospy.get_param('~decel_limit', -5)
        

        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

        # TODO: Add other member variables you need below
    	self.pose = None
    	self.base_waypoints=None
    	self.waypoints_2d = None
        self.waypoint_tree = None
        self.current_velocity = 5
        self.stop_waypoint_indx = -1

        self.loop()
        rospy.spin()
    
    def loop(self):
        rate = rospy.Rate(20)
        while not rospy.is_shutdown():
            if self.pose and self.base_waypoints:
                lane = self.get_lane()
                # rospy.logwarn('waypoints : {0}'.format(len(lane.waypoints)))
                self.final_waypoints_pub.publish(lane)
            rate.sleep()

    # def current_velocity_cb(self, msg):
    #     self.current_velocity = msg.twist.linear.x
	
    def get_lane(self):
        closest_indx = self.get_closest_index()
        lane = Lane()
        lane.header = self.base_waypoints.header
        # lane.waypoints = self.base_waypoints.waypoints[closest_indx:closest_indx+LOOKAHEAD_WPS]
        lane = self.adjust_waypoints_velocity_lane(lane, self.base_waypoints.waypoints, closest_indx)
        return lane

    def adjust_waypoints_velocity_lane(self, lane, waypoints, closest_indx):
        far_point = closest_indx + LOOKAHEAD_WPS
        if self.stop_waypoint_indx == -1 or self.stop_waypoint_indx > far_point:
            lane.waypoints = waypoints[closest_indx:far_point]
            # for i, wp in enumerate(lane.waypoints):
            #     lane.waypoints[i].twist.twist.linear.x = self.current_velocity
            return lane
        # rospy.logwarn('enter the decel')
        temp = []
        #stop_indx = max(self.stop_waypoint_indx-closest_indx-2, 0)
        for i, wp in enumerate(waypoints[closest_indx:far_point]):
            waypoint = Waypoint()
            waypoint.pose = wp.pose
            stop_indx = max(self.stop_waypoint_indx-closest_indx-2, 0)
            dist = self.distance(waypoints[closest_indx:far_point], i, stop_indx)
            vel = math.sqrt(2 * MAX_DECEL*dist)
            if vel < 1.:
                vel = 0
            waypoint.twist.twist.linear.x = vel
            temp.append(waypoint)
        lane.waypoints = temp
        return lane

    def pose_cb(self, msg):
        # TODO: Implement
        self.pose = msg

    def get_closest_index(self):
        x = self.pose.pose.position.x
        y = self.pose.pose.position.y

        closest_indx = self.waypoint_tree.query([x, y], 1)[1]

        closest_point = self.waypoints_2d[closest_indx]
        prev_point = self.waypoints_2d[closest_indx-1]

        # dst_cloest_prev = np.sqrt(closest_point[0]*prev_point[0] + \
        #                     closest_point[1]*prev_point[1])
        # dst_curr_prev = np.sqrt(x*prev_point[0] + \
        #                     y*prev_point[1])
        # if dst_curr_prev <= dst_cloest_prev:
        #     closest_indx = (closest_indx+1) % len(self.waypoints_2d)
        cl_vector = np.array(closest_point)
        prev_vector = np.array(prev_point)
        pos_vector = np.array([x,y])

        val = np.dot((cl_vector - prev_vector), (pos_vector - cl_vector))

        if val > 0:
            closest_indx = (closest_indx + 1) % len(self.waypoints_2d)

        return closest_indx



    def waypoints_cb(self, waypoints):
        # TODO: Implement
        self.base_waypoints = waypoints
        if not self.waypoints_2d:
            self.waypoints_2d = [(point.pose.pose.position.x, point.pose.pose.position.y) for point in waypoints.waypoints]
            self.waypoint_tree = KDTree(self.waypoints_2d)

    def traffic_cb(self, msg):
        # TODO: Callback for /traffic_waypoint message. Implement
        # rospy.logwarn('enter')
        self.stop_waypoint_indx = msg.data

    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        pass

    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x

    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity

    def distance(self, waypoints, wp1, wp2):
        dist = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        for i in range(wp1, wp2+1):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
