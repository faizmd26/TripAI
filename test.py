from tools.tavilty_tool import tavily_search
from tools.flight_tool import search_flights
from backend import run_travel_agent

#res = search_flights("Plan a 10 day trip to RYD  from HYD")
#print(res)
user_input=input("Enter travel request:")

response = run_travel_agent(
    user_input=user_input,
    thread_id="test_user"

)
print("\nFinalresponse")
print(response["answer"])