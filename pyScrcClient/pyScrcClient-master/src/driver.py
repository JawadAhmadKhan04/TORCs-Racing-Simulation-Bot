
import msgParser
import carState
import carControl
import pygame

class Driver(object):
    '''
    A driver object for the SCRC
    '''

    def __init__(self, stage):
        '''Constructor'''
        
        self.logs_data = "data_set.txt"
        with open(self.logs_data, "w") as f:
            f.write("Sensor Log Start\n")
        
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
    
    def drive(self, msg):
        self.state.setFromMsg(msg)
        
        # self.steer()
        
        # self.gear()
        
        # self.speed()
        
        with open(self.logs_data, "a") as f:
            f.write(msg + "\n")

        
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
        