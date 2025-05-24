from controller import Robot
import math

# ===== Robot Initialization =====
robot = Robot()
timestep = int(robot.getBasicTimeStep())  # Simulation timestep (default: 32 ms)

# ----- Ground Sensor Initialization -----
gs = []
for i in range(3):
    gs.append(robot.getDevice('gs'+str(i)))  # Get ground sensor devices: gs0, gs1, gs2
    gs[-1].enable(timestep)                  # Enable each sensor

# ----- Motor Initialization -----
leftMotor = robot.getDevice('left wheel motor')
rightMotor = robot.getDevice('right wheel motor')
leftMotor.setPosition(float('inf'))   # Set to velocity control mode
rightMotor.setPosition(float('inf'))
leftMotor.setVelocity(0.0)            # Initialize velocity to 0
rightMotor.setVelocity(0.0)

# ===== Odometry Setup =====
# Wheel encoder initialization
encoder = []
encoderNames = ['left wheel sensor', 'right wheel sensor']
for i in range(2):
    encoder.append(robot.getDevice(encoderNames[i]))
    encoder[i].enable(timestep)  # Enable encoders

# E-puck physical parameters (in meters)
WHEEL_RADIUS = 0.0205  # Wheel radius (20.5 mm)
WHEEL_BASE = 0.052     # Distance between wheels (53 mm)
MAX_SPEED = 6.28       # Maximum wheel speed (rad/s)

# Odometry variables (start pose: (0, 0.028) facing 90 degrees)
xw = 0.0               # X position in world frame
yw = 0.028             # Y position in world frame (slightly ahead of start line)
alpha = math.pi/2      # Orientation in world frame (radians, 90 degrees)
prev_encoderValues = [0.0, 0.0]  # Previous encoder readings
first_iteration = True           # Flag for first loop iteration

# ===== Control Variables =====
THRESHOLD = 500            # Threshold for line detection (white/black)
all_black_counter = 0      # Counter for consecutive black detections
REQUIRED_BLACK_COUNT = 10  # Number of consecutive black detections to stop
lap_started = False        # Flag indicating lap has started
total_distance = 0.0       # Total distance traveled

print("Starting odometry-based localization...")
print(f"Initial position: xw={xw:.3f}, yw={yw:.3f}, alpha={math.degrees(alpha):.1f}°")

# ===== Main Control Loop =====
while robot.step(timestep) != -1:
    # ----- Read Sensor Data -----
    g = [gsensor.getValue() for gsensor in gs]  # Ground sensor values (0~1000)
    encoderValues = [encoder[0].getValue(), encoder[1].getValue()]  # Encoder readings
    
    # Skip odometry update on the very first iteration
    if first_iteration:
        prev_encoderValues = encoderValues[:]
        first_iteration = False
        continue
    
    # ----- Odometry Calculation -----
    # Calculate wheel movement (radians to meters)
    delta_left = encoderValues[0] - prev_encoderValues[0]
    delta_right = encoderValues[1] - prev_encoderValues[1]
    dl = WHEEL_RADIUS * delta_left   # Left wheel distance
    dr = WHEEL_RADIUS * delta_right  # Right wheel distance
    
    # Calculate robot motion in robot frame
    deltaX = (dl + dr) / 2.0         # Forward movement
    omegaz = (dr - dl) / WHEEL_BASE  # Change in orientation
    
    # Transform incremental motion to world frame
    xw = xw + math.cos(alpha) * deltaX
    yw = yw + math.sin(alpha) * deltaX
    alpha = alpha + omegaz
    
    # Normalize alpha to [-pi, pi]
    while alpha > math.pi:
        alpha -= 2 * math.pi
    while alpha < -math.pi:
        alpha += 2 * math.pi
    
    # Update total distance traveled
    total_distance += abs(deltaX)
    
    # Calculate Euclidean distance from origin (for error measurement)
    error_distance = math.sqrt(xw**2 + yw**2)
    
    # Print odometry and sensor information
    current_time = robot.getTime()
    print(f"Time: {current_time:.1f}s | Position: xw={xw:.3f}, yw={yw:.3f} | "
          f"Orientation: {math.degrees(alpha):.1f}° | "
          f"Distance: {total_distance:.2f}m | GS: [{g[0]:.1f}, {g[1]:.1f}, {g[2]:.1f}]")
    
    # Detect lap start (after moving more than 1 meter)
    if not lap_started and total_distance > 1.0:
        lap_started = True
        print("*** LAP STARTED - Robot has moved 1m+ ***")
    
    # === Improved Stop Condition ===
    # 1. Lap has started (moved sufficiently)
    # 2. Traveled at least 2 meters
    # 3. All sensors detect black for REQUIRED_BLACK_COUNT times in a row
    if (lap_started and 
        total_distance > 2.0 and
        g[0] < THRESHOLD and g[1] < THRESHOLD and g[2] < THRESHOLD):
        
        all_black_counter += 1
        print(f"Stop condition check: {all_black_counter}/{REQUIRED_BLACK_COUNT}")
        
        if all_black_counter >= REQUIRED_BLACK_COUNT:
            print("\n=== STOP CONDITION MET ===")
            print(f"Final position: xw={xw:.3f}, yw={yw:.3f}")
            print(f"Final orientation: {math.degrees(alpha):.1f}°")
            print(f"Final error distance: {error_distance:.3f}m ({error_distance*100:.1f}cm)")
            print(f"Total distance traveled: {total_distance:.2f}m")
            
            # Performance evaluation based on error distance
            if error_distance < 0.15:
                print("EXCELLENT: Error < 15cm!")
            elif error_distance < 0.20:
                print("GOOD: Error < 20cm")
            elif error_distance < 0.25:
                print("ACCEPTABLE: Error < 25cm")
            else:
                print("NEEDS IMPROVEMENT: Error > 25cm")
            
            leftMotor.setVelocity(0.0)
            rightMotor.setVelocity(0.0)
            break
    else:
        all_black_counter = 0  # Reset counter if condition is not met
    
    # Safety: Stop if running too long (over 2 minutes)
    if current_time > 120.0:
        print("\n=== TIME LIMIT REACHED ===")
        print(f"Final position: xw={xw:.3f}, yw={yw:.3f}")
        print(f"Final error: {error_distance:.3f}m")
        leftMotor.setVelocity(0.0)
        rightMotor.setVelocity(0.0)
        break
    
    # ----- Line Following Control Logic -----
    # Center sensor on black, sides on white: go straight
    if (g[0] > 500 and g[1] < 350 and g[2] > 500):
        phildot, phirdot = MAX_SPEED * 0.8, MAX_SPEED * 0.8
    # Right sensor detects black: turn right
    elif (g[2] < 500):
        phildot, phirdot = MAX_SPEED * 0.8, MAX_SPEED * 0
    # Left sensor detects black: turn left
    elif (g[0] < 500):
        phildot, phirdot = MAX_SPEED * 0, MAX_SPEED * 0.8
    # Otherwise: stop
    else:
        phildot, phirdot = MAX_SPEED * 0, MAX_SPEED * 0
    
    # Set motor velocities
    leftMotor.setVelocity(phildot)
    rightMotor.setVelocity(phirdot)
    
    # Update previous encoder values for next iteration
    prev_encoderValues = encoderValues[:]

# ===== Cleanup =====
print("Controller finished.")
