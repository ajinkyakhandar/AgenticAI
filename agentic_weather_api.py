"""Run with: python agentic_weather_api.py"""

from urllib.parse import quote_plus

import requests
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Weather App")


class Question(BaseModel):
    question: str


def get_weather(city: str) -> dict:
    """Get current weather for a city from UWTTR/wttr.in."""
    print(f"Fetching weather for city: {quote_plus(city)}")
    url = f"https://wttr.in/{quote_plus(city)}?format=j1&lang=en"
    try:
        response = requests.get(url, timeout=20)
        response.raise_for_status()
    except requests.RequestException as error:
        raise HTTPException(status_code=502, detail=f"Weather API request failed: {error}") from error

    payload = response.json()
    conditions = payload.get("current_condition") or []
    if not conditions:
        print(f"Weather not found for city: {city}")
        return {"role": "developer", "city": city, "error": f"Weather not found for city: {city}"}
    print(f"Weather data for city {city}: {conditions}")
    current = conditions[0]
    return {
        "role": "developer",
        "city": city,
        "temperature_c": float(current.get("temp_C", 0)),
        "feels_like_c": float(current.get("FeelsLikeC", 0)),
        "humidity_percent": int(current.get("humidity", 0)),
        "weather_desc": current.get("weatherDesc", [{}])[0].get("value", ""),
        "wind_speed_kmh": float(current.get("windspeedKmph", 0)),
    }


@app.post("/weather")
def ask_weather(request: Question) -> dict:
    """Return weather for the city directly."""
    city = request.question.strip()
    if not city:
        raise HTTPException(status_code=400, detail="City name is required.")
    return get_weather(city)


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001)
