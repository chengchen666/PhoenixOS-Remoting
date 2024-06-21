class API:
    def __init__(self, name):
        self.name = name

        # Constants
        self.Payload_forward = 0
        self.Payload_backward = 0
        self.Network_forward = 0
        self.Network_backward = 0
        self.Serialization = 0
        self.Deserialization = 0
        self.Process = 0
        self.Block = 0
        self.Gap = 0

        # Variables
        self.issue_time = 0
        self.start_time = 0
        self.end_time = 0
        self.queue_time = 0
        self.complete_time = 0

    def set_Payload(self, Payload_forward, Payload_backward):
        self.Payload_forward = Payload_forward
        self.Payload_backward = Payload_backward

    def set_Network(self, Network_forward, Network_backward):
        self.Network_forward = Network_forward
        self.Network_backward = Network_backward

    def calc_Network(self, RTT, BANDWIDTH):
        # us, GBps
        BANDWIDTH = BANDWIDTH * 1024 / 1000 * 1024 / 1000 * 1024 # GBps -> Bpus
        self.Network_forward = RTT/2 + \
            float(self.Payload_forward) / BANDWIDTH
        self.Network_backward = RTT/2 + \
            float(self.Payload_backward) / BANDWIDTH

    def set_Serialization(self, Serialization, Deserialization):
        self.Serialization = Serialization
        self.Deserialization = Deserialization

    def set_Process(self, Process):
        self.Process = Process

    def set_Block(self, Block):
        self.Block = Block

    def set_Gap(self, Gap):
        self.Gap = Gap
