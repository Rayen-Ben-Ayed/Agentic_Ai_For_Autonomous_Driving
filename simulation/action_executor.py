import carla
import logging

logger = logging.getLogger(__name__)

class ActionExecutor:
    def __init__(self, carla_client):
        self.carla_client = carla_client
        self.valid_actions = [
            "overtake",
            "follow_lane",
            "stop",
            "yield",
            "change_lane_left",
            "change_lane_right"
        ]

    def execute_action(self, action: str):
        """
        Translates a discrete action into CARLA vehicle controls and applies it to the ego vehicle.
        """
        if action not in self.valid_actions:
            logger.error(f"Invalid action: {action}. Valid actions are: {self.valid_actions}")
            return False

        ego_vehicle = self.carla_client.get_ego_vehicle()
        if not ego_vehicle:
            logger.error("Ego vehicle not initialized. Cannot execute action.")
            return False

        control = carla.VehicleControl()
        
        # Basic mock implementation of discrete actions
        # In a full implementation, this would involve PID controllers or waypoint following
        if action == "follow_lane":
            control.throttle = 0.5
            control.steer = 0.0
            control.brake = 0.0
        elif action == "stop":
            control.throttle = 0.0
            control.steer = 0.0
            control.brake = 1.0
        elif action == "yield":
            # Slow down
            control.throttle = 0.0
            control.steer = 0.0
            control.brake = 0.5
        elif action == "change_lane_left":
            control.throttle = 0.5
            control.steer = -0.3 # Steer left
            control.brake = 0.0
        elif action == "change_lane_right":
            control.throttle = 0.5
            control.steer = 0.3 # Steer right
            control.brake = 0.0
        elif action == "overtake":
            # Speed up and change lane
            control.throttle = 0.8
            control.steer = -0.3
            control.brake = 0.0

        ego_vehicle.apply_control(control)
        logger.info(f"Executed action: {action}")
        return True
