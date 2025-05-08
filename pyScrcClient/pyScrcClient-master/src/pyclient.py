import sys
import argparse
import socket
import driver
import time

if __name__ == '__main__':
    pass

# Configure the argument parser
parser = argparse.ArgumentParser(description='Python client to connect to the TORCS SCRC server.')

parser.add_argument('--host', action='store', dest='host_ip', default='localhost',
                    help='Host IP address (default: localhost)')
parser.add_argument('--port', action='store', type=int, dest='host_port', default=3001,
                    help='Host port number (default: 3001)')
parser.add_argument('--id', action='store', dest='id', default='SCR',
                    help='Bot ID (default: SCR)')
parser.add_argument('--maxEpisodes', action='store', dest='max_episodes', type=int, default=1,
                    help='Maximum number of learning episodes (default: 1)')
parser.add_argument('--maxSteps', action='store', dest='max_steps', type=int, default=0,
                    help='Maximum number of steps (default: 0)')
parser.add_argument('--track', action='store', dest='track', default=None,
                    help='Name of the track')
parser.add_argument('--stage', action='store', dest='stage', type=int, default=3,
                    help='Stage (0 - Warm-Up, 1 - Qualifying, 2 - Race, 3 - Unknown)')

arguments = parser.parse_args()

# Print summary
print('Connecting to server host ip:', arguments.host_ip, '@ port:', arguments.host_port)
print('Bot ID:', arguments.id)
print('Maximum episodes:', arguments.max_episodes)
print('Maximum steps:', arguments.max_steps)
print('Track:', arguments.track)
print('Stage:', arguments.stage)
print('*********************************************')

try:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
except socket.error as msg:
    print('Could not make a socket.')
    sys.exit(-1)

# 20ms timeout
sock.settimeout(0.02)

shutdownClient = False
curEpisode = 0

verbose = False  # Enable for cycle time debugging

d = driver.Driver(arguments.stage)

while not shutdownClient:
    while True:
        print('Sending id to server: ', arguments.id)
        buf = arguments.id + d.init()
        print('Sending init string to server:', buf)
        
        try:
            sock.sendto(buf.encode(), (arguments.host_ip, arguments.host_port))
        except socket.error as msg:
            print("Failed to send data...Exiting...")
            sys.exit(-1)
            
        try:
            buf, addr = sock.recvfrom(1000)
            buf = buf.decode()
        except socket.error as msg:
            if verbose:
                print("Didn't get response from server during init...")
    
        if buf.find('***identified***') >= 0:
            print('Received: ', buf)
            break

    currentStep = 0
    last_time = time.perf_counter()
    
    while True:
        # Enforce 20ms cycle
        current_time = time.perf_counter()
        if (current_time - last_time) * 1000 < 20:
            time.sleep((20 - (current_time - last_time) * 1000) / 1000)
        cycle_time = (current_time - last_time) * 1000
        last_time = current_time
        if verbose:
            print(f"Cycle time: {cycle_time:.2f}ms")

        # Wait for server response
        buf = None
        try:
            buf, addr = sock.recvfrom(1000)
            buf = buf.decode()
        except socket.error as msg:
            if verbose:
                print("Didn't get response from server...")
        
        if verbose and buf:
            print('Received: ', buf)
        
        if buf is not None and buf.find('***shutdown***') >= 0:
            d.onShutDown()
            shutdownClient = True
            print('Client Shutdown')
            break
        
        if buf is not None and buf.find('***restart***') >= 0:
            d.onRestart()
            print('Client Restart')
            break
        
        currentStep += 1
        if currentStep != arguments.max_steps:
            if buf is not None:
                buf = d.drive(buf)
        else:
            buf = '(meta 1)'
        
        if verbose:
            print('Sending: ', buf)
        
        if buf is not None:
            try:
                sock.sendto(buf.encode(), (arguments.host_ip, arguments.host_port))
            except socket.error as msg:
                print("Failed to send data...Exiting...")
                sys.exit(-1)
    
    curEpisode += 1
    
    if curEpisode == arguments.max_episodes:
        shutdownClient = True

sock.close()