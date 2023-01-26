#wensocket client
import asyncio
from warnings import filterwarnings
import websockets

#hide warnings
from generate import Generator
filterwarnings("ignore")

async def test():
    async with websockets.connect('ws://localhost:8000', ping_interval=None, max_size=None) as websocket:
        text = "Console.WriteLine(\"Hello World\");"
        print(f"Sending {text}")
        await websocket.send(text)

        response = await websocket.recv()
        print("********** Response *************")
        print(response)
 
asyncio.get_event_loop().run_until_complete(test())