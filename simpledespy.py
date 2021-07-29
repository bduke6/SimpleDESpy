# Importing random, to use the randint function for event time
import random

'''
When instantiated, the Event class provides the container that holds information about the event. The minimal key 
information includes the time the event should will occur. This time is a numerical value in the form of the simulation 
time. Always at the start of the simulation is time zero. The simulation should continue on until the max time is 
reached. The max time is required as a parameter of the Simulation class (see below). There is no requirement as to the
time period e.g. seconds, minutes, days. This is a design element the analyst must consider when building the model. 

The parameter - agent, is the object that creates the event. Essentially, events are call backs to the object/agent 
letting the agent know that the time has come for the agent to execute some action. So, the agent puts the event in the
simulation and the simulation sends it back when the simulation time meets the time indicated by the event_time 
parameter.

Event_type is not a specific to the Event it is really associated with the Agent. The Agent controls the value 
associated with the event_type. The agent inserts an event into the simulation and then later receives the event back. 
The Agent can have unique types for it specific use. 1 MOVE, 2 SENSE, 3 MESSAGE These are typical values that help 
the agent to process the event. By extending this event one could add objects and data structures as needed. 

The three parameters are the only parameters needed to use the simulation. Additional objects and datastructures could
be considered. For instance, if an agent senses some targets. It could instantiate an event that hold those targets in
a data structure. Put the event into the simulation for immediate callback to the function that processes target
logic and/or the weapon systems that could shoot at the targets.
'''


class Event:

    def __init__(self, event_time, agent, event_type):
        self.event_time = event_time
        self.agent = agent
        self.event_type = event_type

'''


'''
class Agent:
    def __init__(self, uid, sim):
        self.sim = sim
        self.uid = uid
        self.cst = 0  # Current Simulation Time
        self.pos_x = 0
        self.pos_y = 0
        self.speed = 0
        self.dir = 0
        self.tgt_list = []

    # receive_event(event) is a required function for this simulation. Each Agent...in this simulation context, any
    # object that inserts an event onto the simulation queue, is an agent and must have this function.
    def receive_event(self, event):
        t = event.event_type
        self.cst = event.event_time

        if t == 1:
            # MOVE type of event
            self.move_event(event)
        elif t == 2:
            # SENSE type of event
            self.sense_event(event)
        elif t == 3:
            # MESSAGE type of event
            self.message_event(event)

    # Unique to this agent's reaction to a movement event
    def move_event(self, event):
        pass

    # Unique to this agent's reaction to a sense event. This would be like a target in the agent's sensor
    def sense_event(self, event):
        pass

    # Unique to this agent's reaction to a message event. This is good for demonstration. However, a logging of all
    # events should occur in the receive_event function.
    def message_event(self, event):
        print(f"Time: {self.cst} Agent: {self.uid} pos_x: {self.pos_x} pos_y: {self.pos_y}")
        # Note event is passed into to the message_event function. This is so the function can check for any
        # additional information that the Agent might need. Recall that the agent can customize the event as required
        e_t = event.event_type
        if e_t == 3:
            pass
            # In this case the message would be redundant.
            # print(f"{self.uid} just got a type 3 (Message) event.")
        # Calculate the next event that is going to occur. Note that I'm adding on to the current time.
        # For demonstration purposes this agent adds a random number to the current simulation time (cst)
        next_evt_time = self.cst + random.randint(1, 10)
        # Create the event. Note that the event will come back through this agent's receive_event function
        e = Event(event_time=next_evt_time, agent=self, event_type=3)
        self.add_event_to_simulation(e)

    def initial_event(self):
        # Initial event's must be put on queue or else the Agent won't do anything. An agent can have several
        # events on the queue. A good practice would be to put on a move, sense and message event, at the start.
        i_e = Event(1, self, 3)  # Make an initial event
        self.add_event_to_simulation(i_e)  # Add the event to the simulation queue

    def add_event_to_simulation(self, event):
        self.sim.insert(event)  # Passes the event to the simulation



'''


'''
class Simulator:
    def __init__(self, max_time):
        # max simulation time. When an event's time is equal to or greater then this max time the sim will end.
        self.max_t = max_time
        self.queue = []
        self.cst = 0  # Current Simulation Time
        # Creating an "admin or process" simulation event that let's the simulation know when to stop
        # In this case an event that is on the queue until the end. This is a good practice because when building and
        # debugging a model. The queue often goes empty. It is nice to know that there is at least one event on the
        # queue.
        e = Event(event_time=max_time, agent=self, event_type=3)
        self.insert(e)

    def insert(self, e):
        # Appends event to the sim queue
        self.queue.append(e)
        # Assort queue so the event's event time is at the top of the queue
        self.queue.sort(key=lambda event: event.event_time, reverse=True)

    def run(self):
        # Run the simulation loop until the max time or until the queue is empty
        while self.cst <= self.max_t or len(self.queue) == 0:
            # This should pop off the first event on the queue
            e = self.queue.pop()
            # update cst
            self.cst = e.event_time
            # get the agent from the event
            a = e.agent
            # pass the event to the agent for processing
            a.receive_event(e)

    # Like an agent the simulation can put a event into the queue. However, (like the agent) the simulation must have
    # a function called receive_event that accepts an event
    def receive_event(self, event):
        print(f"Max time of: {self.cst} has been reached. Simulation is terminating.")
        # Not a requirement but added in to show potential use
        e_t = event.event_type
        if e_t == 3:
            pass
            # Do something


def main():
    # First you have to make an instance of a simulation
    s = Simulator(max_time=1000)
    # Making two different agents with different unique identification.
    a = Agent(sim=s, uid=f"{0:03}")
    # An agent's initial_event function puts the first Agent events on the simulation
    a.initial_event()  # This puts the first event for this agent on the sim queue
    a = Agent(sim=s, uid=f"{7:03}")
    a.initial_event()  # This puts the first event for this agent on the sim queue
    # Starts the simulation
    s.run()
    # A little test function to see how Python lists get sorted


if __name__ == '__main__':
    main()
