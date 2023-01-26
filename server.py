#websocket server
import asyncio
import json
import socket 
import websockets
from warnings import filterwarnings
import numpy as np

#hide warnings
from generate import Generator
filterwarnings("ignore")
 

#https://stackoverflow.com/a/49677241/1694701
class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

response = []        
# create handler for each connection 
async def handler(websocket, path):     
    consumer_task = asyncio.ensure_future(
        consumer_handler(websocket, path))
    producer_task = asyncio.ensure_future(
        producer_handler(websocket, path))
    done, pending = await asyncio.wait(
        [consumer_task, producer_task],
        return_when=asyncio.FIRST_COMPLETED,
    )
    for task in pending:
        task.cancel()

async def consumer_handler(websocket, path):
    async for message in websocket:
        consumer(message)


def consumer(data):
    global response
    print(f"\n{data}\n")
    try:
        result = gen.generate(data,random_seed=True)
        response = result
    except:
        response = ["nada"] 

async def producer_handler(websocket, path):
    global response
    while True:
        message = await producer()  
        #print(f"sending {type(message)}")      
        result = message # json.dumps(message, cls=NumpyEncoder)
        await websocket.send(result)
        response = []

async def producer():
    while len(response) == 0:
        await asyncio.sleep(1)        
    return response
 
if __name__=="__main__": 
    hostname=socket.gethostname()   
    IPAddr=socket.gethostbyname(hostname)   
    print("Your Computer Name is:"+hostname)   
    print("Your Computer IP Address is:"+IPAddr) 
    print("Starting C_Sharp_nano_GPT server.....") 
    start_server = websockets.serve(handler, "localhost", 8000) 
    gen = Generator()
    print("Listening....")
    asyncio.get_event_loop().run_until_complete(start_server) 
    asyncio.get_event_loop().run_forever()