import msgParser
import carState
import carControl
import pygame
import csv
import time
import os
import re

class Driver(object):
    '''
    A driver object for the SCRC
    '''

    def __init__(self, stage):
        '''Constructor'''
        # self.logs_data_csv = "dataset.csv"
        # if not os.path.exists(self.logs_data_csv):
        #     with open(self.logs_data_csv, "w", newline="") as f:
        #         csv_writer = csv.writer(f)
        #         csv_writer.writerow([
        #             "timestamp", "angle", "curLapTime", "damage", "distFromStart", "distRaced",
        #             "fuel", "gear", "lastLapTime", "racePos", "rpm", "speedX", "speedY", "speedZ",
        #             "trackPos", "z", "focus_0", "focus_1", "focus_2", "focus_3", "focus_4",
        #             "track_0", "track_1", "track_2", "track_3", "track_4", "track_5", "track_6",
        #             "track_7", "track_8", "track_9", "track_10", "track_11", "track_12", "track_13",
        #             "track_14", "track_15", "track_16", "track_17", "track_18",
        #             "opponent_0", "opponent_1", "opponent_2", "opponent_3", "opponent_4",
        #             "opponent_5", "opponent_6", "opponent_7", "opponent_8", "opponent_9",
        #             "opponent_10", "opponent_11", "opponent_12", "opponent_13", "opponent_14",
        #             "opponent_15", "opponent_16", "opponent_17", "opponent_18", "opponent_19",
        #             "opponent_20", "opponent_21", "opponent_22", "opponent_23", "opponent_24",
        #             "opponent_25", "opponent_26", "opponent_27", "opponent_28", "opponent_29",
        #             "opponent_30", "opponent_31", "opponent_32", "opponent_33", "opponent_34",
        #             "opponent_35",
        #             "wheelSpinVel_0", "wheelSpinVel_1", "wheelSpinVel_2", "wheelSpinVel_3",
        #             "accel", "brake", "clutch", "steer", "inputs",
        #             "closest_opponent_distance", "closest_opponent_direction", "relative_opponent_speed",
        #             "minimum_track_distance", "track_curvature", "track_width", "num_nearby_opponents",
        #             "car_acceleration", "steering_rate", "overtaking_opportunity", "reward"
        #         ])
        pygame.init()
        
        self.gear_change_delay = 500
        self.last_gear_change = 0
        self.use_sensor_steering = False  # Enable sensor-based steering
        
        self.WARM_UP = 0
        self.QUALIFYING = 1
        self.RACE = 2
        self.UNKNOWN = 3
        self.stage = stage
        
        self.parser = msgParser.MsgParser()
        self.state = carState.CarState()
        self.control = carControl.CarControl()
        
        self.steer_lock = 0.785398
        self.max_speed = 100
        self.prev_rpm = None
        self.prev_speedX = 0
        self.prev_steer = 0
        self.prev_opponents = [200] * 36
        self.prev_closest_distance = 200
        
        self.acceleration = 0.1
        self.steering_movement = 0.1
        self.braking = 0.3
        self.smoothing_rate = 0.15
        
        # Optimize Pygame
        pygame.display.set_mode((100, 100))  # Smaller window
        pygame.display.set_caption("Controller")
        pygame.event.set_allowed([pygame.QUIT, pygame.KEYDOWN, pygame.KEYUP])
    
    def init(self):
        '''Return init string with rangefinder angles'''
        self.angles = [0 for x in range(19)]
        for i in range(5):
            self.angles[i] = -90 + i * 15
            self.angles[18 - i] = 90 - i * 15
        for i in range(5, 9):
            self.angles[i] = -20 + (i-5) * 5
            self.angles[18 - i] = 20 - (i-5) * 5
        return self.parser.stringify({'init': self.angles})
    
    def parse_message(self, msg):
        """Parses sensor data and returns a structured list."""
        try:
            matches = re.findall(r"\((\w+)((?:\s-?\d+\.?\d*e?-?\d*)+)\)", msg)
            parsed_data = {key: value.strip().split() for key, value in matches}

            for key in parsed_data:
                if len(parsed_data[key]) == 1:
                    parsed_data[key] = parsed_data[key][0]

            timestamp = time.perf_counter()  # Higher precision

            accel = self.control.getAccel()
            brake = self.control.getBrake() if hasattr(self.control, 'getBrake') else 0
            gear = self.control.getGear()
            clutch = self.control.getClutch() if hasattr(self.control, 'getClutch') else 0
            steer = self.control.getSteer()
            inputs = 0

            angle = float(parsed_data.get("angle", 0))
            curLapTime = float(parsed_data.get("curLapTime", 0))
            damage = float(parsed_data.get("damage", 0))
            distFromStart = float(parsed_data.get("distFromStart", 0))
            distRaced = float(parsed_data.get("distRaced", 0))
            fuel = float(parsed_data.get("fuel", 0))
            lastLapTime = float(parsed_data.get("lastLapTime", 0))
            racePos = float(parsed_data.get("racePos", 0))
            rpm = float(parsed_data.get("rpm", 0))
            speedX = float(parsed_data.get("speedX", 0))
            speedY = float(parsed_data.get("speedY", 0))
            speedZ = float(parsed_data.get("speedZ", 0))
            trackPos = float(parsed_data.get("trackPos", 0))
            z = float(parsed_data.get("z", 0))

            focus = [float(x) for x in parsed_data.get("focus", [0]*5)[:5]]
            focus = focus + [0] * (5 - len(focus))

            track = [float(x) for x in parsed_data.get("track", [0]*19)[:19]]
            track = track + [0] * (19 - len(track))

            opponents = [float(x) for x in parsed_data.get("opponents", [200]*36)[:36]]
            opponents = opponents + [200] * (36 - len(opponents))

            wheelSpinVel = [float(x) for x in parsed_data.get("wheelSpinVel", [0]*4)[:4]]
            wheelSpinVel = wheelSpinVel + [0] * (4 - len(wheelSpinVel))

            # New features
            closest_opponent_distance = min(opponents)
            closest_opponent_direction = opponents.index(closest_opponent_distance) if closest_opponent_distance < 200 else -1
            relative_opponent_speed = 0
            if closest_opponent_distance < 200 and self.prev_closest_distance < 200:
                delta_time = 0.02  # 20ms cycle
                relative_opponent_speed = (self.prev_closest_distance - closest_opponent_distance) / delta_time
            minimum_track_distance = min(track)
            track_curvature = (track[0] - track[18]) / (track[0] + track[18] + 1e-6)  # Left vs. right
            track_width = track[4] + track[14]  # Approximate width at ±45°
            num_nearby_opponents = sum(1 for x in opponents if x < 50)
            car_acceleration = (speedX - self.prev_speedX) / 0.02
            steering_rate = (steer - self.prev_steer) / 0.02
            overtaking_opportunity = 1 if (closest_opponent_distance < 20 and 
                                         abs(closest_opponent_direction - 0) <= 5 and 
                                         track_width > 15) else 0
            reward = (1 - racePos / 10) - damage / 1000 - abs(trackPos) / 2
            if closest_opponent_distance < 50:
                reward += (self.prev_closest_distance - closest_opponent_distance) * 0.1

            row = [
                timestamp, angle, curLapTime, damage, distFromStart, distRaced,
                fuel, gear, lastLapTime, racePos, rpm, speedX, speedY, speedZ,
                trackPos, z
            ] + focus + track + opponents + wheelSpinVel + [
                accel, brake, clutch, steer, inputs,
                closest_opponent_distance, closest_opponent_direction, relative_opponent_speed,
                minimum_track_distance, track_curvature, track_width, num_nearby_opponents,
                car_acceleration, steering_rate, overtaking_opportunity, reward
            ]

            # Update previous values
            self.prev_speedX = speedX
            self.prev_steer = steer
            self.prev_opponents = opponents
            self.prev_closest_distance = closest_opponent_distance

            return row

        except Exception as e:
            print(f"Error parsing message: {e}")
            return None
    
    def drive(self, msg):
        self.state.setFromMsg(msg)
        
        if self.use_sensor_steering:
            self.steer()
        
        parsed_data = self.parse_message(msg)
        # if parsed_data:
        #     with open(self.logs_data_csv, "a", newline="") as f:
        #         csv_writer = csv.writer(f)
        #         csv_writer.writerow(parsed_data)
        
        self.controls()
        
        return self.control.toMsg()

    def controls(self):
        pygame.event.pump()  # Efficient event processing
        keys = pygame.key.get_pressed()

        if keys[pygame.K_LEFT]:
            new_steer = self.control.getSteer() + self.steering_movement
            self.control.setSteer(min(1.0, new_steer))
        elif keys[pygame.K_RIGHT]:
            new_steer = self.control.getSteer() - self.steering_movement
            self.control.setSteer(max(-1.0, new_steer))
        else:
            pass
            current = self.control.getSteer()
            if current < 0:
                self.control.setSteer(min(0, current + self.smoothing_rate))
            elif current > 0:
                self.control.setSteer(max(0, current - self.smoothing_rate))

        if keys[pygame.K_DOWN]:
            if hasattr(self.control, 'setBrake'):
                self.control.setBrake(self.braking)
            self.control.setAccel(0)
        elif keys[pygame.K_UP]:
            new_accel = self.control.getAccel() + self.acceleration
            self.control.setAccel(min(1.0, new_accel))
            if hasattr(self.control, 'setBrake'):
                self.control.setBrake(0)
        else:
            current_accel = self.control.getAccel()
            if current_accel < 0:
                self.control.setAccel(min(0, current_accel + self.acceleration))
            elif current_accel > 0:
                self.control.setAccel(max(0, current_accel - self.acceleration))
            if hasattr(self.control, 'setBrake'):
                self.control.setBrake(0)

        current_gear = self.state.getGear()
        current_time = pygame.time.get_ticks()

        if current_time - self.last_gear_change >= self.gear_change_delay:
            if keys[pygame.K_z]:
                new_gear = min(current_gear + 1, 6)
                self.control.setGear(new_gear)
                self.last_gear_change = current_time
            if keys[pygame.K_x]:
                new_gear = max(current_gear - 1, -1)
                self.control.setGear(new_gear)
                self.last_gear_change = current_time

        return self.control.toMsg()
    
    def steer(self):
        angle = self.state.getAngle()
        dist = self.state.getTrackPos()
        sensor_steer = (angle - dist * 0.5) / self.steer_lock
        current_steer = self.control.getSteer()
        blended_steer = 0.7 * current_steer + 0.3 * sensor_steer
        self.control.setSteer(max(-1.0, min(1.0, blended_steer)))
    
    def gear(self):
        rpm = self.state.getRpm()
        gear = self.state.getGear()
        
        if self.prev_rpm is None:
            up = True
        else:
            up = (self.prev_rpm - rpm) < 0
        
        if up and rpm > 7000:
            gear += 1
        if not up and rpm < 3000:
            gear -= 1
        
        self.control.setGear(gear)
    
    def speed(self):
        speed = self.state.getSpeedX()
        accel = self.control.getAccel()
        
        if speed < self.max_speed:
            accel += 0.1
            if accel > 1:
                accel = 1.0
        else:
            accel -= 0.1
            if accel < 0:
                accel = 0.0
        
        self.control.setAccel(accel)
    
    def onShutDown(self):
        print("Shutting down")
        pygame.quit()
    
    def onRestart(self):
        print("Restarting")
        pass