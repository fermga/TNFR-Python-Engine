import asyncio
import aiohttp
import json

async def test_position_volume_query():
    url = "https://gmx.squids.live/gmx-synthetics-arbitrum:prod/api/graphql"
    query = """
    {
      positionsVolume(
        accountsIds: ["*"]
      ) {
        volume
        volumeInUsd
        timestamp
        marketId
        accountId
      }
    }
    """
    
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json={'query': query}) as response:
            print(f"Status: {response.status}")
            if response.status == 200:
                data = await response.json()
                print(json.dumps(data, indent=2))
            else:
                print(await response.text())

if __name__ == "__main__":
    asyncio.run(test_position_volume_query())