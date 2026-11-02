from tavily import TavilyClient
import os
from dotenv import load_dotenv

load_dotenv()

client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

def tavily_search(query):
    response = client.search(query=query,max_results=5)

    results = []

    for i, r in enumerate(response['results'],1):
        title = r.get('title', 'No Title')
        url = r.get('url', 'No URL')
        snippet = r.get('snippet', 'No Snippet')

        if len(snippet) > 300:
            snippet = snippet[:300].rsplit(' ', 1)[0] + '...'

        results.append(f"{i}. {title}\nURL: {url}\nSnippet: {snippet}\n")
    return "\n".join(results)