
import msgParser
import carState
import carControl
import pygame
import csv
import os
import re

class Driver(object):
    '''
    A driver object for the SCRC
    '''

    def __init__(self, stage):
        '''Constructor'''
        
        self.logs_data = "data_set.txt"
        if not os.path.exists(self.logs_data):
            with open(self.logs_data, "w") as f:
                f.write("Sensor Log Start\n")
            
        self.logs_data_csv = "data_set.csv"
        if not os.path.exists(self.logs_data_csv):
            with open(self.logs_data_csv, "w", newline="") as f:
                csv_writer = csv.writer(f)
                csv_writer.writerow(["angle", "curLapTime", "damage", "distFromStart", "distRaced", 
                                    "fuel", "gear", "lastLapTime", "racePos", "rpm", "speedX", "speedY", 
                                    "speedZ", "trackPos", "z"] + 
                                    [f"opponent_{i}" for i in range(36)] + 
                                    [f"track_{i}" for i in range(19)] + 
                                    [f"wheelSpinVel_{i}" for i in range(4)] + 
                                    [f"focus_{i}" for i in range(5)])
        pygame.init()
        
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

            # Extract specific keys
            angle = parsed_data.get("angle", 0)
            curLapTime = parsed_data.get("curLapTime", 0)
            damage = parsed_data.get("damage", 0)
            distFromStart = parsed_data.get("distFromStart", 0)
            distRaced = parsed_data.get("distRaced", 0)
            fuel = parsed_data.get("fuel", 0)
            gear = parsed_data.get("gear", 0)
            lastLapTime = parsed_data.get("lastLapTime", 0)
            racePos = parsed_data.get("racePos", 0)
            rpm = parsed_data.get("rpm", 0)
            speedX = parsed_data.get("speedX", 0)
            speedY = parsed_data.get("speedY", 0)
            speedZ = parsed_data.get("speedZ", 0)
            trackPos = parsed_data.get("trackPos", 0)
            z = parsed_data.get("z", 0)

            # Extract list values (ensure they have correct lengths)
            opponents = parsed_data.get("opponents", [200]*36)[:36]
            track = parsed_data.get("track", [0]*19)[:19]
            wheelSpinVel = parsed_data.get("wheelSpinVel", [0]*4)[:4]
            focus = parsed_data.get("focus", [-1]*5)[:5]

            # Flatten the data into a row
            row = [angle, curLapTime, damage, distFromStart, distRaced, fuel, gear, lastLapTime, 
                   racePos, rpm, speedX, speedY, speedZ, trackPos, z] + \
                   opponents + track + wheelSpinVel + focus

            return row

        except Exception as e:
            print(f"Error parsing message: {e}")
            return None
    
    def drive(self, msg):
        self.state.setFromMsg(msg)
        
        # self.steer()
        
        # self.gear()
        
        # self.speed()
        
        with open(self.logs_data, "a") as f:
            f.write(msg + "\n")

        parsed_data = self.parse_message(msg)
        if parsed_data:
            with open(self.logs_data_csv, "a", newline="") as f:
                csv_writer = csv.writer(f)
                csv_writer.writerow(parsed_data)
        
        self.controls()
        
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

        if keys[pygame.K_z]:  # Gear Up
            new_gear = min(current_gear + 1, 6)  # Max gear limit
            self.control.setGear(new_gear)

        if keys[pygame.K_x]:  # Gear Down
            new_gear = max(current_gear - 1, 1)  # Min gear limit
            self.control.setGear(new_gear)

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
        pass
    
    def onRestart(self):
        print("Restarting")
        pass
        