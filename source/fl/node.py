

class Node:
    class Config:
        def __init__(self, id: str, children: list, parent: str):
            self.id = id
            self.children = children
            self.parent = parent
            

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