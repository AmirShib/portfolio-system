from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import yfinance as yf
import json
import asyncio
from redis import asyncio as aioredis
from src.core.config import settings
from src.core.backtester import BacktestEngine
from contextlib import asynccontextmanager

# Global State
redis_pool = None
UNIVERSE = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global redis_pool, UNIVERSE
    print("Startup: Connecting to Redis...")
    redis_pool = aioredis.ConnectionPool.from_url(
        f"redis://{settings.REDIS_HOST}:{settings.REDIS_PORT}", 
        decode_responses=True
    )
    
    print("Startup: Loading Market Universe...")
    
    data = yf.download(
        settings.SECTORS + [settings.BENCHMARK], 
        period="5y", 
        threads=True, 
        auto_adjust=True
    )
    
    if data.empty:
        print("Critical Error: Yahoo Finance returned no data.")
        raise RuntimeError("Market Data Download Failed")
    
    UNIVERSE = data["Close"]
    
    print(f"Data Loaded successfully. Assets: {len(UNIVERSE.columns)}")
    
    yield
    
    print("Shutdown: Closing Redis...")
    await redis_pool.disconnect()

app = FastAPI(title="Portfolio Backend", lifespan=lifespan)

class SimRequest(BaseModel):
    asset: str
    strategy: str
    params: dict

@app.post("/simulate")
async def simulate(req: SimRequest):
    if req.asset not in UNIVERSE.columns:
        raise HTTPException(400, "Asset not in universe")
        
    key = f"sim:{BacktestEngine.generate_key(req.asset, req.strategy, req.params)}"
    redis = aioredis.Redis(connection_pool=redis_pool)
    
    # Check Cache
    cached = await redis.get(key)
    if cached:
        return json.loads(cached)
        
    try:
        prices = UNIVERSE[req.asset].dropna()
        # Run VectorBT in a thread to keep API responsive
        res = await asyncio.to_thread(BacktestEngine.run, prices, req.strategy, req.params)
        
        await redis.set(key, json.dumps(res), ex=86400)
        return res
    except Exception as e:
        raise HTTPException(500, str(e))
    finally:
        await redis.close()