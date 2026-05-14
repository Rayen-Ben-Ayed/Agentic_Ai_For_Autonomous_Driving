class BaseScenario:
    def __init__(self, carla_client):
        self.carla_client = carla_client
        self.world = carla_client.get_world()
        self.npc_actors = []
        self.llm_queried = False # Flag to ensure we don't spam the LLM

    def setup(self):
        """
        Spawn NPC vehicles, pedestrians, or change weather.
        To be implemented by child classes.
        """
        pass

    def is_llm_needed(self, world_state):
        """
        Check the world state to see if the critical event has happened.
        Returns True if the LLM should be asked for a decision.
        """
        return False

    def teardown(self):
        """
        Clean up all NPCs created for this scenario.
        """
        for actor in self.npc_actors:
            if actor.is_alive:
                actor.destroy()
        self.npc_actors.clear()
