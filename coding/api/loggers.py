import httpx


class CallCountManager:
    def __init__(self, url, key):
        self.url = url
        self.key = key
        self.headers = {"Content-Type": "application/json"}

    async def add(self):
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.url}/counter/add",
                params={"api_key": self.key},
                headers=self.headers,
            )
            response.raise_for_status()
            return response.json()
