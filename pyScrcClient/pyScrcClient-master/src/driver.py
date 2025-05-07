import msgParser
import carState
import carControl
import pygame
import csv
import time
import os
import re
import datetime
import joblib
import numpy as np

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
        #             "accel", "brake", "clutch", "steer", "inputs"
        #         ])
        pygame.init()
        
        self.gear_change_delay = 500
        self.last_gear_change = 0  # Store last gear change time

        
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
        
        self.acceleration = 0.1  
        self.steering_movement = 0.1  
        self.braking = 0.3      

        # Load trained model and scaler
        self.model = joblib.load('torcs_mlp_model.joblib')
        self.scaler = joblib.load('torcs_scaler.joblib')
        # Define model input features
        self.model_features = [
            'angle', 'rpm', 'speedX', 'speedY', 'speedZ', 'trackPos'
        ] + [f'track_{i}' for i in range(19)]
        
        pygame.display.set_mode((300, 300))
        pygame.display.set_caption("Controller")
    
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

            # Convert single-value lists to scalar
            for key in parsed_data:
                if len(parsed_data[key]) == 1:
                    parsed_data[key] = parsed_data[key][0]

            # Get current timestamp
            timestamp = datetime.datetime.now().isoformat()

            # Extract control values
            accel = self.control.getAccel()
            brake = self.control.getBrake() if hasattr(self.control, 'getBrake') else 0
            gear = self.control.getGear()
            clutch = self.control.getClutch() if hasattr(self.control, 'getClutch') else 0
            steer = self.control.getSteer()
            inputs = 0  # Placeholder for inputs

            # Extract sensor values
            angle = parsed_data.get("angle", 0)
            curLapTime = parsed_data.get("curLapTime", 0)
            damage = parsed_data.get("damage", 0)
            distFromStart = parsed_data.get("distFromStart", 0)
            distRaced = parsed_data.get("distRaced", 0)
            fuel = parsed_data.get("fuel", 0)
            lastLapTime = parsed_data.get("lastLapTime", 0)
            racePos = parsed_data.get("racePos", 0)
            rpm = parsed_data.get("rpm", 0)
            speedX = parsed_data.get("speedX", 0)
            speedY = parsed_data.get("speedY", 0)
            speedZ = parsed_data.get("speedZ", 0)
            trackPos = parsed_data.get("trackPos", 0)
            z = parsed_data.get("z", 0)

            # Extract focus values
            focus = parsed_data.get("focus", [0]*5)[:5]
            focus = focus + [0] * (5 - len(focus))  # Pad with zeros if needed

            # Extract track values
            track = parsed_data.get("track", [0]*19)[:19]
            track = track + [0] * (19 - len(track))  # Pad with zeros if needed

            # Extract opponent values
            opponents = parsed_data.get("opponents", [200]*36)[:36]
            opponents = opponents + [200] * (36 - len(opponents))  # Pad with 200 if needed

            # Extract wheel spin velocities
            wheelSpinVel = parsed_data.get("wheelSpinVel", [0]*4)[:4]
            wheelSpinVel = wheelSpinVel + [0] * (4 - len(wheelSpinVel))  # Pad with zeros if needed

            # Create row with all values in the specified order
            row = [
                timestamp, angle, curLapTime, damage, distFromStart, distRaced,
                fuel, gear, lastLapTime, racePos, rpm, speedX, speedY, speedZ,
                trackPos, z
            ] + focus + track + opponents + wheelSpinVel + [
                accel, brake, clutch, steer, inputs
            ]

            return row

        except Exception as e:
            print(f"Error parsing message: {e}")
            return None
    
    def get_model_input(self):
        """Extracts the model input features from the current car state."""
        state = self.state
        features = [
            float(getattr(state, 'angle', 0)),
            float(getattr(state, 'rpm', 0)),
            float(getattr(state, 'speedX', 0)),
            float(getattr(state, 'speedY', 0)),
            float(getattr(state, 'speedZ', 0)),
            float(getattr(state, 'trackPos', 0)),
        ]
        # Add track sensors
        if hasattr(state, 'track') and len(state.track) == 19:
            features += [float(x) for x in state.track]
        else:
            features += [0.0]*19
        return features
    
    def drive(self, msg):
        self.state.setFromMsg(msg)
        
        # Use model to predict controls
        model_input = np.array(self.get_model_input()).reshape(1, -1)
        model_input_scaled = self.scaler.transform(model_input)
        accel, brake, steer = self.model.predict(model_input_scaled)[0]
        self.control.setAccel(float(accel))
        if hasattr(self.control, 'setBrake'):
            self.control.setBrake(float(brake))
        self.control.setSteer(float(steer))
        
        # parsed_data = self.parse_message(msg)
        # if parsed_data:
        #     with open(self.logs_data_csv, "a", newline="") as f:
        #         csv_writer = csv.writer(f)
        #         csv_writer.writerow(parsed_data)
        
        # No manual controls, only model
        # self.controls()
        
        return self.control.toMsg()

    def controls(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pass

        keys = pygame.key.get_pressed()

        # Adjust steering with less sensitivity using left/right arrow keys.
        if keys[pygame.K_LEFT]:
            new_steer = self.control.getSteer() + self.steering_movement
            self.control.setSteer(min(1.0, new_steer))
        elif keys[pygame.K_RIGHT]:
            new_steer = self.control.getSteer() - self.steering_movement
            self.control.setSteer(max(-1.0, new_steer))
        else:
            # Gradually return steering to center when no key is pressed.
            current = self.control.getSteer()
            if current < 0:
                self.control.setSteer(min(0, current + self.steering_movement))
            elif current > 0:
                self.control.setSteer(max(0, current - self.steering_movement))
                

        # Adjust acceleration and braking.
        if keys[pygame.K_DOWN]:
            if hasattr(self.control, 'setBrake'):
                self.control.setBrake(self.braking)
            self.control.setAccel(0)
        elif keys[pygame.K_UP]:
            # Increase acceleration slowly, ensuring it does not exceed 1.0.
            new_accel = self.control.getAccel() + self.acceleration
            self.control.setAccel(min(1.0, new_accel))
            # Reset brake if accelerating
            if hasattr(self.control, 'setBrake'):
                self.control.setBrake(0)
        else:
            # When neither up nor down are pressed, gradually reduce acceleration.
            current_accel = self.control.getAccel()
            if current_accel < 0:
                self.control.setAccel(min(0, current_accel + self.acceleration))
            elif current_accel > 0:
                self.control.setAccel(max(0, current_accel - self.acceleration))
            if hasattr(self.control, 'setBrake'):
                self.control.setBrake(0)

# Manual gear shifting
        current_gear = self.state.getGear()
        current_time = pygame.time.get_ticks()  # Get current time in milliseconds

        if current_time - self.last_gear_change >= self.gear_change_delay:
            if keys[pygame.K_z]:  # Gear Up
                new_gear = min(current_gear + 1, 6)  # Max gear limit
                self.control.setGear(new_gear)
                self.last_gear_change = current_time  # Update last change time

            if keys[pygame.K_x]:  # Gear Down
                new_gear = max(current_gear - 1, -1)  # Min gear limit
                self.control.setGear(new_gear)
                self.last_gear_change = current_time  # Update last change time

        

        
        # if keys[pygame.K_z]:  # Gear Up
        #     new_gear = min(current_gear + 1, 6)  # Max gear limit
        #     self.control.setGear(new_gear)

        # if keys[pygame.K_x]:  # Gear Down
        #     new_gear = max(current_gear - 1, 1)  # Min gear limit
        #     self.control.setGear(new_gear)

        return self.control.toMsg()

        
    
    def steer(self):
        angle = self.state.angle
        dist = self.state.trackPos
        
        self.control.setSteer((angle - dist*0.5)/self.steer_lock)
    
    def gear(self):
        rpm = self.state.getRpm()
        gear = self.state.getGear()
        
        if self.prev_rpm == None:
            up = True
        else:
            if (self.prev_rpm - rpm) < 0:
                up = True
            else:
                up = False
        
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
        