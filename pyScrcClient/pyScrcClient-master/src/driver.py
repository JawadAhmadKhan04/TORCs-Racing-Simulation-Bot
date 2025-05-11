import msgParser
import carState
import carControl
import csv
import time
import os
import re
import numpy as np
import joblib
import pandas as pd

class Driver(object):
    '''
    A driver object for the SCRC
    '''

    def __init__(self, stage):
        '''Constructor'''
        import joblib

        # Load models
        self.models = {
            "accel": joblib.load("xgb_model_accel.pkl"),
            "brake": joblib.load("xgb_model_brake.pkl"),
            "steer": joblib.load("xgb_model_steer.pkl"),
            "gear": joblib.load("xgb_model_gear.pkl")
        }

        # Load input scaler
        self.scaler = joblib.load("xgb_scaler.pkl")

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
                
            phase = "straight"
            if speedX < 5 and distRaced < 10:
                phase = "start"
            elif abs(trackPos) > 1 and (speedX > 10 or abs(angle) > 45):
                phase = "recovery"
            elif overtaking_opportunity == 1:
                phase = "overtaking"
            elif abs(angle) > 20 or abs(track_curvature) > 0.1:
                phase = "corner"

            row = [
                timestamp, angle, curLapTime, damage, distFromStart, distRaced,
                fuel, gear, lastLapTime, racePos, rpm, speedX, speedY, speedZ,
                trackPos, z
            ] + focus + track + opponents + wheelSpinVel + [
                accel, brake, clutch, steer, inputs,
                closest_opponent_distance, closest_opponent_direction, relative_opponent_speed,
                minimum_track_distance, track_curvature, track_width, num_nearby_opponents,
                car_acceleration, steering_rate, overtaking_opportunity, reward, phase
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

        try:
            # === Safely get sensor values ===
            speedX = self.state.getSpeedX() or 0.0
            speedY = self.state.getSpeedY() or 0.0
            speedZ = self.state.getSpeedZ() or 0.0
            angle = self.state.getAngle() or 0.0
            trackPos = self.state.getTrackPos() or 0.0
            distFromStart = self.state.getDistFromStart() or 0.0
            distRaced = self.state.getDistRaced() or 0.0
            racePos = self.state.getRacePos() or 0.0
            rpm = self.state.getRpm() or 0.0
            lastLapTime = self.state.getLastLapTime() or 0.0
            curLapTime = self.state.getCurLapTime() or 0.0
            steer = self.control.getSteer() or 0.0
            

            # print(f"distFromStart: {distFromStart}")

            track = self.state.getTrack()
            if track is None or len(track) != 19:
                track = [0.0] * 19

            opponents = self.state.getOpponents()
            if opponents is None or len(opponents) != 36:
                opponents = [200.0] * 36

            closest_opponent_distance = min(opponents)
            closest_opponent_direction = opponents.index(closest_opponent_distance) if closest_opponent_distance < 200 else -1
            minimum_track_distance = min(track)
            track_curvature = (track[0] - track[18]) / (track[0] + track[18] + 1e-6)  # Left vs. right
            track_width = track[4] + track[14]  # Approximate width at ±45°
            num_nearby_opponents = sum(1 for x in opponents if x < 50)
            car_acceleration = (speedX - self.prev_speedX) / 0.02
            steering_rate = (steer - self.prev_steer) / 0.02
            overtaking_opportunity = 1 if (closest_opponent_distance < 20 and 
                                         abs(closest_opponent_direction - 0) <= 5 and 
                                         track_width > 15) else 0
            damage = self.state.getDamage() or 0.0
            reward = (1 - racePos / 10) - damage / 1000 - abs(trackPos) / 2
            if closest_opponent_distance < 50:
                reward += (self.prev_closest_distance - closest_opponent_distance) * 0.1
            
            phase = "straight"
            if speedX < 5 and distRaced < 10:
                phase = "start"
            elif abs(trackPos) > 1 and (speedX > 10 or abs(angle) > 45):
                phase = "recovery"
            elif overtaking_opportunity == 1:
                phase = "overtaking"
            elif abs(angle) > 20 or abs(track_curvature) > 0.1:
                phase = "corner"

            # Create one-hot encoded phase
            phase_df = pd.DataFrame({'phase': [phase]})
            phase_dummies = pd.get_dummies(phase_df['phase'], prefix='phase')
            
            # Load saved phase categories and ensure all categories are present
            phase_categories = joblib.load("phase_categories.pkl")
            for category in phase_categories:
                if category not in phase_dummies.columns:
                    phase_dummies[category] = 0
            
            # Reorder columns to match training data
            phase_dummies = phase_dummies[phase_categories]
            phase_encoded = phase_dummies.values[0]

            # === Build feature vector ===
            features = [speedX, speedY, speedZ, angle, trackPos, distFromStart, distRaced, rpm, lastLapTime, curLapTime, 
                       closest_opponent_distance, closest_opponent_direction, minimum_track_distance, track_curvature, 
                       track_width, num_nearby_opponents, car_acceleration, steering_rate, overtaking_opportunity, reward] + \
                       list(phase_encoded) + track + opponents
            
            input_vector = np.array(features).reshape(1, -1)
            input_vector = self.scaler.transform(input_vector)

            # === Predict each control action ===
            accel = float(np.clip(self.models["accel"].predict(input_vector)[0], 0.0, 1.0))
            brake = float(np.clip(self.models["brake"].predict(input_vector)[0], 0.0, 1.0))
            steer = float(np.clip(self.models["steer"].predict(input_vector)[0], -1.0, 1.0))
            gear_raw = self.models["gear"].predict(input_vector)[0]
            gear = int(np.clip(round(gear_raw), -1, 6))  # Clamp to valid gear range

            # === Set control actions ===
            self.control.setAccel(accel)
            self.control.setBrake(brake)
            self.control.setSteer(steer)
            self.control.setGear(gear)

        except Exception as e:
            print(f"[ML-Control Error] {e} — fallback to emergency brake.")
            self.control.setAccel(0)
            self.control.setBrake(1)
            self.control.setSteer(0)
            self.control.setGear(1)

        return self.control.toMsg()

    def onShutDown(self):
        print("Shutting down")
        # pygame.quit()
    
    def onRestart(self):
        print("Restarting")
        pass