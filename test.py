import asyncio
# from mcp_client_test import get_all_tools , tavily_mcp_search 
from mcp_client import get_all_tools, tavily_mcp_search


if __name__ == "__main__":
   # query= "latest new about ai"
    asyncio.run(get_all_tools())