class API:
    def __init__(self, name):
        self.name = name

        # Constants
        self.Network_forward = 0
        self.Network_backward = 0
        self.Process = 0
        self.Block = 0
        self.Gap = 0

        # Variables
        self.issue_time = 0
        self.start_time = 0
        self.end_time = 0
        self.queue_time = 0
        self.complete_time = 0

    def set_Network(self, Network_forward, Network_backward):
        self.Network_forward = Network_forward
        self.Network_backward = Network_backward

    def set_Process(self, Process):
        self.Process = Process

    def set_Block(self, Block):
        self.Block = Block

    def set_Gap(self, Gap):
        self.Gap = Gap
