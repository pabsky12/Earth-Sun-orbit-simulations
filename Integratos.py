def normal_euler(old_position, old_velocity, acceleration, time_step):
    new_position = [old_position[0] + (old_velocity[0] * time_step), old_position[1] + (old_velocity[1] * time_step)] 
    new_velocity = [old_velocity[0] + (acceleration[0] * time_step), old_velocity[1] + (acceleration[1] * time_step)]

    return new_position, new_velocity

def euler_cromer(old_position, old_velocity, acceleration, time_step):
    new_velocity = [old_velocity[0] + (acceleration[0] * time_step), old_velocity[1] + (acceleration[1] * time_step)]
    new_position = [old_position[0] + (new_velocity[0] * time_step), old_position[1] + (new_velocity[1]* time_step)]

    return new_position, new_velocity

simulators = [normal_euler, euler_cromer]