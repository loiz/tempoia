import asyncio
import sys
import os
from tempoia import TempoWeatherPredictor

# Mock settings if needed, or just rely on defaults
# We need to make sure we can instantiate the predictor

async def main():
    print("Initializing predictor...")
    predictor = TempoWeatherPredictor()
    
    print("Running benchmark...")
    results = predictor.benchmark_algorithms()
    
    print("\nBenchmark Results:")
    import json
    print(json.dumps(results, indent=2))
    
    if results and "_best" in results:
        print(f"\nBest Algorithm: {results['_best']['key']}")
    else:
        print("\nNo best algorithm found.")

if __name__ == "__main__":
    asyncio.run(main())
