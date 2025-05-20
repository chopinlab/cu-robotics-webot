from controller import Robot

# Initialize simulation
robot = Robot()
timestep = int(robot.getBasicTimeStep())

# E-puck robot rotation constants (simulation-based adjustment values)
# Using rotationStep value observed from the robot model
ROTATION_STEP = 0.262  # Rotation step (radians)
MOTOR_SPEED = 3.0      # Motor speed (radians/sec)

# Directly specify rotation times (experimentally determined values)
TIME_FOR_180_DEGREES = 3  # seconds (180 degree rotation)
TIME_FOR_90_DEGREES = 1.5  # seconds (90 degree rotation)

print(f"Rotation step (rotationStep): {ROTATION_STEP} radians")
print(f"Time needed for 180 degree rotation: {TIME_FOR_180_DEGREES:.2f} seconds")
print(f"Time needed for 90 degree rotation: {TIME_FOR_90_DEGREES:.2f} seconds")

# FSM state definitions
STATE_FORWARD = "FORWARD"
STATE_U_TURN = "U_TURN"
STATE_APPROACH_SECOND_OBSTACLE = "APPROACH_SECOND_OBSTACLE"
STATE_ROTATE_CLOCKWISE = "ROTATE_CLOCKWISE"
STATE_DRIVE_FORWARD_UNTIL_STOP = "DRIVE_FORWARD_UNTIL_STOP"
STATE_STOP = "STOP"

# Initial state and timer setup
current_state = STATE_FORWARD
turn_start_time = None  # Record rotation start time

# Initialize sensors
distance_sensors = []
sensor_names = ['ps0', 'ps1', 'ps2', 'ps3', 'ps4', 'ps5', 'ps6', 'ps7']
for name in sensor_names:
    sensor = robot.getDevice(name)
    sensor.enable(timestep)
    distance_sensors.append(sensor)

# Initialize motors - using names from the robot model
left_motor = robot.getDevice('left wheel motor')
right_motor = robot.getDevice('right wheel motor')
left_motor.setPosition(float('inf'))
right_motor.setPosition(float('inf'))

# Function to set motor speeds
def set_speed(left, right):
    left_motor.setVelocity(left)
    right_motor.setVelocity(right)

# FSM operation
while robot.step(timestep) != -1:
    # Read sensor values
    sensor_values = [sensor.getValue() for sensor in distance_sensors]
    
    # Check front sensors (ps0, ps1, ps7)
    front_sensor_values = [sensor_values[0], sensor_values[1], sensor_values[7]]
    front_obstacle = any(value > 200 for value in front_sensor_values)
    
    # Check left sensor (ps5)
    left_sensor_value = sensor_values[5]
    
    # Check rear sensors (ps3, ps4)
    rear_sensor_values = [sensor_values[3], sensor_values[4]]
    rear_obstacle = any(value > 80 for value in rear_sensor_values)
    
    # FSM state-specific operations
    if current_state == STATE_FORWARD:
        set_speed(MOTOR_SPEED, MOTOR_SPEED)  # Move forward
        if front_obstacle:  # First obstacle detected
            print("Front obstacle detected! Starting U-turn.")
            current_state = STATE_U_TURN
            turn_start_time = robot.getTime()  # Record rotation start time
    
    elif current_state == STATE_U_TURN:
        # Time-based 180 degree rotation
        set_speed(-MOTOR_SPEED/2, MOTOR_SPEED/2)  # Left turn
        
        if robot.getTime() - turn_start_time >= TIME_FOR_180_DEGREES:
            print(f"U-turn complete after {TIME_FOR_180_DEGREES:.2f} seconds! Proceeding to second obstacle.")
            set_speed(0, 0)  # Briefly stop to stabilize
            robot.step(timestep * 10)  # Wait for 10 steps
            current_state = STATE_APPROACH_SECOND_OBSTACLE
    
    elif current_state == STATE_APPROACH_SECOND_OBSTACLE:
        set_speed(MOTOR_SPEED, MOTOR_SPEED)  # Move forward
        if front_obstacle:  # Second obstacle detected
            print("Second obstacle detected! Starting clockwise rotation.")
            current_state = STATE_ROTATE_CLOCKWISE
            turn_start_time = robot.getTime()  # Record rotation start time
    
    elif current_state == STATE_ROTATE_CLOCKWISE:
        # Time-based 90 degree rotation (clockwise)
        set_speed(MOTOR_SPEED/2, -MOTOR_SPEED/2)  # Right turn
        
        # Complete rotation when left sensor detects wall or time elapses
        if left_sensor_value > 300:
            print(f"Left wall detected during rotation! Sensor value: {left_sensor_value:.2f}")
            set_speed(0, 0)  # Briefly stop to stabilize
            robot.step(timestep * 5)  # Wait for 5 steps
            current_state = STATE_DRIVE_FORWARD_UNTIL_STOP
        elif robot.getTime() - turn_start_time >= TIME_FOR_90_DEGREES:
            print(f"Clockwise rotation complete after {TIME_FOR_90_DEGREES:.2f} seconds!")
            set_speed(0, 0)  # Briefly stop to stabilize
            robot.step(timestep * 5)  # Wait for 5 steps
            current_state = STATE_DRIVE_FORWARD_UNTIL_STOP
    
    elif current_state == STATE_DRIVE_FORWARD_UNTIL_STOP:
        if left_sensor_value < 200:  # Wall no longer detected on the left
            print(f"Left sensor no longer detects obstacle. Stopping. Value: {left_sensor_value:.2f}")
            current_state = STATE_STOP
        else:
            print(f"Left sensor detects obstacle. Driving forward. Value: {left_sensor_value:.2f}")
            set_speed(MOTOR_SPEED, MOTOR_SPEED)  # Continue moving forward
    
    elif current_state == STATE_STOP:
        set_speed(0.0, 0.0)  # Stop
        print("Robot has stopped.")
    
    # Print debug information (every 500ms)
    if int(robot.getTime() * 2) % 1 == 0:
        print(f"Current State: {current_state}")
        print(f"Front Obstacle: {front_obstacle}, Left Sensor Value: {left_sensor_value:.2f}")
        if current_state in [STATE_U_TURN, STATE_ROTATE_CLOCKWISE] and turn_start_time is not None:
            elapsed_time = robot.getTime() - turn_start_time
            target_time = TIME_FOR_180_DEGREES if current_state == STATE_U_TURN else TIME_FOR_90_DEGREES
            print(f"Rotation Time: {elapsed_time:.2f}s / Target: {target_time:.2f}s ({elapsed_time/target_time*100:.1f}%)")