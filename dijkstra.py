#Based on the implementation found:
#https://dev.to/mxl/dijkstras-algorithm-in-python-algorithms-for-beginners-dkc

############################################
############################################

#TODO:
#1. create topology
#2. create timestamp objects
# a) add the timestamps and 

#3. calculate EDT
#4. remove the edges used
#5. Add link creation with probability p0

#+1: always have the updated info
#+2: create Jupyter notebook

#Are the packets sent simultanously, or one after the other?

############################################
############################################

from collections import deque, namedtuple
import asyncio, sched, time, random, logging
from datetime import datetime

logging.basicConfig(filename='dijkstra.log',level=logging.DEBUG)
logging.debug('This is the logfile for the Dijkstra routing algorithm on the quantum network at %s', time.time)

# we'll use infinity as a default distance to nodes.
inf = float('inf')
Edge = namedtuple('Edge', 'start, end, cost, capacity')

Nodes = {
		1 : "a",
		2 : "b",
		3 : "c",
		4 : "d",
		5 : "e",
		6 : "f",
        7 : "g",
        8 : "h"
	}

#defining a global threshold time and capacity for each of the channels
#will most probably vary in the future
timeThreshold = 100
originalCapacity = 1
originalCost = 1

def make_edge(start, end, cost=originalCost, capacity=originalCapacity):
    return Edge(start, end, cost, capacity)

def make_backward_edge(start, end, cost=originalCost, capacity=originalCapacity):
    return Edge(end, start, cost, capacity)

#TODO:
#remove potential duplicate paths
#TODO:
#recreate this part using an iterator
def gen_rand_pairs(number):
    result = []
    for x in range(number):
        source = random.randint(1,number)
        dest = random.randint(1,number)
        while source == dest:
            dest = random.randint(1,number)
        result += [[Nodes[source],Nodes[dest]]]
    return result

class Graph:
    s = sched.scheduler(time.time, time.sleep)
    def __init__(self, edges):
        # let's check that the data is right
        wrong_edges = [i for i in edges if len(i) not in [2, 4]]
        if wrong_edges:
            raise ValueError('Wrong edges data: %s', wrong_edges)

        self.edges = [make_edge(*edge) for edge in edges] + [make_backward_edge(*edge) for edge in edges]

    # @property:
    # https://www.programiz.com/python-programming/property 
    @property
    def vertices(self):
        return set(
            sum(
                ([edge.start, edge.end] for edge in self.edges), []
            )
        )

    def add_link(self, startNode, endNode, cost=originalCost, capacity=originalCapacity):
        node_pairs = [[startNode, endNode], [endNode, startNode]]
        edges = self.edges[:]
        for edge in edges:
            if [edge.start, edge.end] in node_pairs:
                self.edges.remove(edge)

                #Add bidirectional link
                self.edges.append(Edge(start=startNode, end=endNode, cost=cost, capacity=capacity))
                self.edges.append(Edge(start=endNode, end=startNode, cost=cost, capacity=capacity))

    async def rebuild_link(self, startNode, endNode, cost=originalCost, capacity=originalCapacity):
        await rebuilding(startNode, endNode)
        self.add_link(startNode, endNode, cost, capacity)
        logging.debug('Link between %s and %s has been rebuilt.', startNode, endNode)

    def remove_link(self, startNode, endNode):
        
        node_pairs = [[startNode, endNode], [endNode, startNode]]
        edges = self.edges[:]
        for edge in edges:
            if [edge.start, edge.end] in node_pairs:
                
                #We need to wait if currently there is no available link through the channel
                #if edge.capacity == 0:
                    
                #we have used a channel along this edge, remove one edge from here
                self.edges.remove(edge)
                self.edges.append(make_edge(edge.start, edge.end,edge.cost,edge.capacity-1))

    def get_link_capacity(self, startNode, endNode):
        node_pairs = [[startNode, endNode], [endNode, startNode]]
        edges = self.edges[:]
        for edge in edges:
            if [edge.start, edge.end] in node_pairs:
                return edge.capacity
    
    async def process_step(self, startNode, endNode, edt):

        #If the capacity of the link is 0, then add the waiting time for rebuilding
        if self.get_link_capacity(startNode,endNode)==0:
            edt += timeThreshold  

        #Remove the link between startNode and endNode
        self.remove_link(startNode,endNode)

        #Initializing the rebuild
        self.rebuild_link(startNode,endNode,originalCost,1)
        await self.rebuild_link(startNode,endNode,originalCost,1)

        #Incrementing the entanglement delay time
        edt += 1
        return edt


    async def process_path(self, currentPath):
        
        #Initializing entanglement delay time and idleness parameter
        edt = 0

        #Take the leftmost two nodes out of the deque and get the edt until we are finished
        while True:
            edt += self.process_step(currentPath.popleft(),currentPath.popleft(),edt)
        if (currentPath)==0:
            logging.debug('This path took %s long.', edt)        
            return edt

    @property
    def neighbours(self):
        neighbours = {vertex: set() for vertex in self.vertices}
        for edge in self.edges:
            neighbours[edge.start].add((edge.end, edge.cost))

        return neighbours

    def dijkstra(self, source, dest):
        
        #Checking if the given source is existing or not
        assert source in self.vertices, 'Such source node doesn\'t exist'

        #Running the Initialize-Single-Source procedure
        distances = {vertex: inf for vertex in self.vertices}
        previous_vertices = {
            vertex: None for vertex in self.vertices
        }
        distances[source] = 0
        vertices = self.vertices.copy()

        #picking each vertex out of the min-priority deque
        while vertices:
            current_vertex = min(
                vertices, key=lambda vertex: distances[vertex])
            vertices.remove(current_vertex)
            
            #Relaxation procedure
            if distances[current_vertex] == inf:
                break
            for neighbour, cost in self.neighbours[current_vertex]:
                alternative_route = distances[current_vertex] + cost
                if alternative_route < distances[neighbour]:
                    distances[neighbour] = alternative_route
                    previous_vertices[neighbour] = current_vertex

        path, current_vertex = deque(), dest
        while previous_vertices[current_vertex] is not None:
            path.appendleft(current_vertex)
            current_vertex = previous_vertices[current_vertex]
        if path:
            path.appendleft(current_vertex)
        return path

#Defining the graph
#Vertex names: a-h
graph = Graph([
       
    #Initialize circular graph
    ("a", "b"),  ("b", "c"),   ("c", "d"), ("d", "e"), ("e", "f"), ("f", "g"), ("g", "h"), ("h", "a"),


    #Add "diagonal" edges
    ("a", "c"), ("d", "e"), ("e", "g"),  ("g", "a")])

graphOriginalEdges = graph.edges.copy()

#TODO
#Procedure for the probabilistic link creation

#TODO
#Procedure for the link creation: both probabilistic and threshold based
#Uses the Graph add_link procedure

#Generate random pairs of nodes between which we are seeking a path
randPairs = gen_rand_pairs(8)

#Assemble paths into one deque
paths = deque()
for pair in randPairs:
    paths.appendleft(graph.dijkstra(pair[0],pair[1]))

logging.debug('%s', paths)

async def rebuilding(startNode, endNode):  
    logging.debug('Rebuilding the link between nodes %s and %s at %s', startNode, endNode, datetime.now())
    await asyncio.sleep(timeThreshold)

async def send_packets(graph, paths):
    while True:
        #Process the next path in the deque
        currentPath = paths.popleft()
        graph.process_path(currentPath)
        if(len(paths)==0):
            break
    
        

start = time.time()  
loop = asyncio.get_event_loop()

tasks = [  
    asyncio.ensure_future(send_packets(graph, paths))
]
loop.run_until_complete(asyncio.wait(tasks))  
loop.close()

end = time.time()  
logging.debug("Total time: %s", (end - start))
logging.debug(graph.edges)