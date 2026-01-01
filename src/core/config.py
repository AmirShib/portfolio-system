import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Secrets (These must be in your .env file)
    SUPABASE_URL: str
    SUPABASE_KEY: str
    
    # Infrastructure
    REDIS_HOST: str = "redis"
    REDIS_PORT: int = 6379
    
    # Financial Universe (V1: Sectors + Benchmark)
    SECTORS: list = ["XLK", "XLV", "XLF", "XLC", "XLY", "XLP", "XLE", "XLI", "XLB", "XLRE", "XLU"]
    BENCHMARK: str = "SPY"
    
    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()