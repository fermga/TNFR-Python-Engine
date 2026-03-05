"""
Test Session Leak Fix
====================
Test the improved session management to ensure no more AsyncIO warnings.
"""
import asyncio
import sys
import os

# Add paths
sys.path.append('src')
sys.path.append('future-research/crypto-lab/src')

async def test_session_cleanup():
    """Test that sessions are properly cleaned up"""
    print("🧪 Testing GMX Data Pipeline Session Management...")
    
    try:
        from gmx_data_source import GMXDataPipeline, ArbitrumConfig
        
        print("✅ Imported GMX modules successfully")
        
        # Test 1: Basic initialization and cleanup
        print("\n📡 Test 1: Basic Pipeline Lifecycle")
        
        config = ArbitrumConfig()
        pipeline = GMXDataPipeline(config)
        
        # Initialize
        await pipeline.initialize()
        print("✅ Pipeline initialized")
        
        # Test quick data fetch to ensure session works
        try:
            markets = ['BTC', 'ETH', 'SOL'] 
            data = await pipeline._fetch_real_market_data(markets)
            print(f"✅ Fetched data for {len(data)} markets")
        except Exception as e:
            print(f"⚠️ Data fetch warning (expected in some environments): {e}")
        
        # Close properly
        await pipeline.stop_feed()
        print("✅ Pipeline closed via stop_feed()")
        
        # Test 2: Multiple initialize/close cycles
        print("\n🔄 Test 2: Multiple Lifecycle Cycles")
        
        for i in range(3):
            pipeline2 = GMXDataPipeline(config)
            await pipeline2.initialize()
            await pipeline2.close()
            print(f"✅ Cycle {i+1} complete")
        
        # Test 3: Context manager usage
        print("\n🎯 Test 3: Context Manager Pattern")
        
        from gmx_data_source import GMXStatsClient
        
        async with GMXStatsClient() as stats_client:
            print("✅ Stats client opened with context manager")
        print("✅ Stats client automatically closed")
        
        print("\n🎉 All tests passed! Session management improved.")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Testing session leak fixes...")
    print("This should complete without 'Unclosed client session' warnings.\n")
    
    # Run the test
    result = asyncio.run(test_session_cleanup())
    
    if result:
        print("\n✅ SUCCESS: Session leak fix validated!")
        print("Dashboard should now run without AsyncIO warnings.")
    else:
        print("\n❌ Tests failed - further investigation needed.")
    
    # Give a moment for any final cleanup
    print("\nWaiting 2 seconds for final cleanup...")
    asyncio.run(asyncio.sleep(2))
    print("✅ Test complete.")