
# depict the hardware structure of fl
# how the devices are connected to each other

class Node:
    class Config:
        def __init__(self, id: str, neighbors: list, tasks: list):
            self.id = id
            self.neighbors = neighbors
            
    def __init__(self, config: Config):
        self.config = config
    
    def update(self):
        pass
    

class Trainer(Node):
    def __init__(self, config: Node.Config):
        super().__init__(config)

class Aggregator(Node):
    def __init__(self, config: Node.Config):
        super().__init__(config)