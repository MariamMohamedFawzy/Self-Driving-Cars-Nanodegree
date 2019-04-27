
GAS_DENSITY = 2.858
ONE_MPH = 0.44704

from yaw_controller import YawController
from pid import PID
from lowpass import LowPassFilter
import rospy


class Controller(object):
    def __init__(self, *args, **kwargs):
        self.vehicle_mass = args[0]
        self.fuel_capacity = args[1]
        self.brake_deadband = args[2]
        self.decel_limit = args[3]
        self.accel_limit = args[4]
        self.wheel_radius = args[5]
        self.wheel_base = args[6]
        self.steer_ratio = args[7]
        self.max_lat_accel = args[8]
        self.max_steer_angle = args[9]

        self.yaw_controller = YawController(self.wheel_base, self.steer_ratio, 0.1,
        					 self.max_lat_accel, self.max_steer_angle)
        
        kp = 0.3
        ki = 0.1
        kd = 0.
        mn = 0.
        mx = .2
        self.throttle_controller = PID(kp, ki, kd, mn=mn, mx=mx)

        tau = 0.5
        ts = 0.02
        self.low_pass = LowPassFilter(tau, ts)

        self.last_time = rospy.get_time()
        # self.last_vel = None



   #  def control(self, *args, **kwargs):
   #      # TODO: Change the arg, kwarg list to suit your needs
   #      # Return throttle, brake, steer
   #      twist_linear = args[0]
   #      twist_angular = args[1]
   #      current_velocity = args[2]
   #      dbw_enabled = args[3]


   #      # if self.last_vel is None:
   #      # 	self.last_vel = current_velocity


   #      if not dbw_enabled:
   #      	self.throttle_controller.reset()
   #      	return 0., 0., 0.

   #      current_velocity = self.low_pass.filt(current_velocity)

   #      steering = self.yaw_controller.get_steering(twist_linear, 
   #      			twist_angular, current_velocity)

   #      vel_err = twist_linear - current_velocity
   #      self.last_vel = current_velocity
        	
   #      current_time = rospy.get_time()
   #      diff_time = current_time - self.last_time
   #      self.last_time = current_time

   #      throttle = self.throttle_controller.step(vel_err, diff_time)
   #      brake = 0
   #      rospy.logwarn('twist_linear = {0}, current_velocity = {1}'.format(twist_linear, current_velocity))
   #      if twist_linear == 0 and current_velocity < 0.1:
			# throttle = 0
			# brake = 700
   #      elif throttle < 0.1 and vel_err < 0:
			# throttle = 0
			# decel = max(vel_err, self.decel_limit)
			# brake = abs(decel) * self.vehicle_mass * self.wheel_radius

   #      return throttle, brake, steering


    def control(self, *args, **kwargs):
        # TODO: Change the arg, kwarg list to suit your needs
        # Return throttle, brake, steer
        twist_linear = args[0]
        twist_angular = args[1]
        current_velocity = args[2]
        dbw_enabled = args[3]


        # if self.last_vel is None:
        #   self.last_vel = current_velocity


        if not dbw_enabled:
            self.throttle_controller.reset()
            return 0., 0., 0.

        current_velocity = self.low_pass.filt(current_velocity)

        steering = self.yaw_controller.get_steering(twist_linear, 
                    twist_angular, current_velocity)

        vel_err = twist_linear - current_velocity
        # self.last_vel = current_velocity
            
        current_time = rospy.get_time()
        diff_time = current_time - self.last_time
        self.last_time = current_time

        throttle = self.throttle_controller.step(vel_err, diff_time)
        # rospy.logwarn('vel_err = {3}, throttle = {2}, twist_linear = {0}, current_velocity = {1}'.format(twist_linear, current_velocity, throttle, vel_err))
        # brake = 0
        # if throttle < 0.1:
        #     decel = max(vel_err, self.decel_limit)
        #     brake = abs(decel) * self.vehicle_mass * self.wheel_radius

        brake = 0
        # rospy.logwarn('twist_linear = {0}, current_velocity = {1}'.format(twist_linear, current_velocity))
        if twist_linear == 0. and current_velocity < 0.1:
            throttle = 0
            brake = 700
        elif throttle < 0.1 and vel_err < 0.:
            throttle = 0
            decel = max(vel_err, self.decel_limit)
            brake = abs(decel) * self.vehicle_mass * self.wheel_radius

        return throttle, brake, steering


