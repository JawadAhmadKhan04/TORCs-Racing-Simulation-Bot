import msgParser
import carState
import carControl
import pygame
import csv
import time
import os
import re
import torch
import joblib
import numpy as np

class TorcsDrivingModel(torch.nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=[512, 256, 128, 64]):
        super(TorcsDrivingModel, self).__init__()
        layers = []
        prev_dim = input_dim
        for dim in hidden_dims:
            layers.extend([
                torch.nn.Linear(prev_dim, dim),
                torch.nn.BatchNorm1d(dim),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.15)
            ])
            prev_dim = dim
        layers.append(torch.nn.Linear(prev_dim, output_dim))
        self.network = torch.nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.network(x)
        steer = torch.tanh(x[:, 0:1])
        accel_brake = torch.sigmoid(x[:, 1:3])
        gear = x[:, 3:4] * 7.0 - 1.0
        return torch.cat([steer, accel_brake, gear], dim=1)

class Driver(object):
    def __init__(self, stage):
        self.logs_data_csv = "dataset.csv"
        self.actions_log_csv = "actions_log.csv"
        if not os.path.exists(self.logs_data_csv):
            with open(self.logs_data_csv, "w", newline="") as f:
                csv_writer = csv.writer(f)
                csv_writer.writerow([
                    "timestamp", "angle", "curLapTime", "damage", "distFromStart", "distRaced",
                    "fuel", "gear", "lastLapTime", "racePos", "rpm", "speedX", "speedY", "speedZ",
                    "trackPos", "z", "focus_0", "focus_1", "focus_2", "focus_3", "focus_4",
                    "track_0", "track_1", "track_2", "track_3", "track_4", "track_5", "track_6",
                    "track_7", "track_8", "track_9", "track_10", "track_11", "track_12", "track_13",
                    "track_14", "track_15", "track_16", "track_17", "track_18",
                    "opponent_0", "opponent_1", "opponent_2", "opponent_3", "opponent_4",
                    "opponent_5", "opponent_6", "opponent_7", "opponent_8", "opponent_9",
                    "opponent_10", "opponent_11", "opponent_12", "opponent_13", "opponent_14",
                    "opponent_15", "opponent_16", "opponent_17", "opponent_18", "opponent_19",
                    "opponent_20", "opponent_21", "opponent_22", "opponent_23", "opponent_24",
                    "opponent_25", "opponent_26", "opponent_27", "opponent_28", "opponent_29",
                    "opponent_30", "opponent_31", "opponent_32", "opponent_33", "opponent_34",
                    "opponent_35",
                    "wheelSpinVel_0", "wheelSpinVel_1", "wheelSpinVel_2", "wheelSpinVel_3",
                    "accel", "brake", "clutch", "steer", "inputs",
                    "closest_opponent_distance", "closest_opponent_direction", "relative_opponent_speed",
                    "minimum_track_distance", "track_curvature", "track_width", "num_nearby_opponents",
                    "car_acceleration", "steering_rate", "overtaking_opportunity", "reward", "phase"
                ])
        if not os.path.exists(self.actions_log_csv):
            with open(self.actions_log_csv, "w", newline="") as f:
                csv_writer = csv.writer(f)
                csv_writer.writerow(["timestamp", "steer", "accel", "brake", "gear", "trackPos", "speedX", "angle", "distRaced", "overtaking_opportunity", "phase"])
        
        pygame.init()
        window_width, window_height = 200, 200
        screen_info = pygame.display.Info()
        screen_width = screen_info.current_w
        screen_height = screen_info.current_h
        x_pos = max(0, screen_width - window_width - 50)
        y_pos = max(0, min(50, screen_height - window_height - 50))
        os.environ['SDL_VIDEO_WINDOW_POS'] = f"{x_pos},{y_pos}"
        pygame.display.set_mode((window_width, window_height))
        pygame.display.set_caption("TORCS Controller")
        pygame.event.set_allowed([pygame.QUIT, pygame.KEYDOWN, pygame.KEYUP])
        print("Pygame window opened in top-right corner. Click it to focus for manual control.")
        
        self.gear_change_delay = 500
        self.last_gear_change = 0
        self.use_sensor_steering = False
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
        self.use_model = False
        self.device = torch.device("cpu")
        self.input_features = [
            "angle", "trackPos", "speedX", "speedY", "speedZ",
            "rpm", "distFromStart", "distRaced", "racePos",
            "closest_opponent_distance", "closest_opponent_direction",
            "minimum_track_distance", "track_curvature", "track_width",
            "num_nearby_opponents", "overtaking_opportunity"
        ] + [f"track_{i}" for i in range(19)] + [f"opponent_{i}" for i in range(36)]
        if self.use_model:
            try:
                self.model = TorcsDrivingModel(input_dim=71, output_dim=4).to(self.device)
                self.model.load_state_dict(torch.load("torcs_model.pth", map_location=self.device, weights_only=True))
                self.model.eval()
                self.scaler = joblib.load("scaler.pkl")
                print("Loaded torcs_model.pth and scaler.pkl for autonomous driving.")
            except Exception as e:
                print(f"Error loading model or scaler: {e}")
                self.use_model = False
    
    def init(self):
        self.angles = [0 for x in range(19)]
        for i in range(5):
            self.angles[i] = -90 + i * 15
            self.angles[18 - i] = 90 - i * 15
        for i in range(5, 9):
            self.angles[i] = -20 + (i-5) * 5
            self.angles[18 - i] = 20 - (i-5) * 5
        return self.parser.stringify({'init': self.angles})
    
    def parse_message(self, msg):
        try:
            matches = re.findall(r"\((\w+)((?:\s-?\d+\.?\d*e?-?\d*)+)\)", msg)
            parsed_data = {key: value.strip().split() for key, value in matches}
            for key in parsed_data:
                if len(parsed_data[key]) == 1:
                    parsed_data[key] = parsed_data[key][0]
            timestamp = time.perf_counter()
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
            closest_opponent_distance = min(opponents)
            closest_opponent_direction = opponents.index(closest_opponent_distance) if closest_opponent_distance < 200 else -1
            relative_opponent_speed = 0
            if closest_opponent_distance < 200 and self.prev_closest_distance < 200:
                delta_time = 0.02
                relative_opponent_speed = (self.prev_closest_distance - closest_opponent_distance) / delta_time
            minimum_track_distance = min(track)
            track_curvature = (track[0] - track[18]) / (track[0] + track[18] + 1e-6)
            track_width = track[4] + track[14]
            num_nearby_opponents = sum(1 for x in opponents if x < 50)
            car_acceleration = (speedX - self.prev_speedX) / 0.02
            steering_rate = (steer - self.prev_steer) / 0.02
            overtaking_opportunity = 1 if (closest_opponent_distance < 20 and 
                                         abs(closest_opponent_direction - 0) <= 5 and 
                                         track_width > 15) else 0
            reward = (1 - racePos / 10) - damage / 1000 - abs(trackPos) / 2
            if closest_opponent_distance < 50:
                reward += (self.prev_closest_distance - closest_opponent_distance) * 0.1
            # Determine phase
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
            self.prev_speedX = speedX
            self.prev_steer = steer
            self.prev_opponents = opponents
            self.prev_closest_distance = closest_opponent_distance
            return row
        except Exception as e:
            print(f"Error parsing message: {e}")
            return None
    
    def drive(self, msg):
        try:
            self.state.setFromMsg(msg)
            parsed_data = self.parse_message(msg)
            if parsed_data:
                with open(self.logs_data_csv, "a", newline="") as f:
                    csv_writer = csv.writer(f)
                    csv_writer.writerow(parsed_data)
            
            if self.use_model and parsed_data:
                state = np.array([parsed_data[self.input_features.index(f)] for f in self.input_features])
                state = self.scaler.transform(state.reshape(1, -1))
                state_tensor = torch.FloatTensor(state).to(self.device)
                with torch.no_grad():
                    actions = self.model(state_tensor).cpu().numpy().flatten()
                steer = actions[0]  # Removed * 1.2 scaling
                accel = actions[1]
                brake = actions[2]
                gear = round(max(-1, min(6, actions[3])))
                # Race start override
                speedX = parsed_data[self.input_features.index("speedX")]
                trackPos = parsed_data[self.input_features.index("trackPos")]
                distRaced = parsed_data[self.input_features.index("distRaced")]
                angle = parsed_data[self.input_features.index("angle")]
                phase = parsed_data[-1]
                if phase == "start" or (speedX < 10 and distRaced < 20):
                    gear = 1
                    accel = 0.8
                    brake = 0.0
                    steer = 0.0
                # Straight phase override
                elif phase == "straight" and abs(angle) < 10 and abs(trackPos) <= 1:
                    steer = max(-0.1, min(0.1, steer))  # Constrain steer for straight driving
                # Mutual exclusivity
                if accel > 0.5:
                    brake = 0.0
                elif brake > 0.5:
                    accel = 0.0
                # Recovery adjustments
                if abs(trackPos) > 1 and (speedX > 10 or abs(angle) > 45):
                    brake = 0.5
                    accel = 0.0
                    gear = -1
                    steer = steer * 0.5  # Dampen steer in recovery
                # Ensure valid ranges
                steer = max(-1.0, min(1.0, steer))
                accel = max(0.0, min(1.0, accel))
                brake = max(0.0, min(1.0, brake))
                self.control.setSteer(steer)
                self.control.setAccel(accel)
                self.control.setBrake(brake)
                self.control.setGear(gear)
                # Log actions
                with open(self.actions_log_csv, "a", newline="") as f:
                    csv_writer = csv.writer(f)
                    csv_writer.writerow([
                        time.perf_counter(), steer, accel, brake, gear, trackPos,
                        speedX, angle, distRaced, parsed_data[self.input_features.index("overtaking_opportunity")], phase
                    ])
                # print(f"Actions: steer={steer:.4f}, accel={accel:.4f}, brake={brake:.4f}, gear={gear}, speedX={speedX:.2f}, angle={angle:.2f}, distRaced={distRaced:.2f}, phase={phase}")
            else:
                self.controls()
            
            return self.control.toMsg()
        except Exception as e:
            print(f"Error in drive: {e}")
            self.controls()
            return self.control.toMsg()

    def controls(self):
        pygame.event.pump()
        keys = pygame.key.get_pressed()
        current_steer = self.control.getSteer()
        if keys[pygame.K_LEFT]:
            new_steer = current_steer + self.steering_movement
            self.control.setSteer(min(1.0, new_steer))
        elif keys[pygame.K_RIGHT]:
            new_steer = current_steer - self.steering_movement
            self.control.setSteer(max(-1.0, new_steer))
        else:
            # Gradually reset steer to 0 when no steering keys are pressed
            if current_steer > 0:
                new_steer = max(0.0, current_steer - self.steering_movement)
            else:
                new_steer = min(0.0, current_steer + self.steering_movement)
            self.control.setSteer(new_steer)
        if keys[pygame.K_DOWN]:
            if hasattr(self.control, 'setBrake'):
                self.control.setBrake(self.braking)
            self.control.setAccel(0)
        elif keys[pygame.K_UP]:
            new_acc = self.control.getAccel() + self.acceleration
            self.control.setAccel(min(1.0, new_acc))
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
        # Reduce current_steer weight to prevent stickiness
        blended_steer = 0.3 * current_steer + 0.7 * sensor_steer
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