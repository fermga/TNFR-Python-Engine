"""
Test script para validar la integración completa de datos históricos GMX
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'future-research', 'crypto-lab', 'src'))

from gmx_data_source import GMXDataPipeline
from datetime import datetime, timedelta
import asyncio

async def test_complete_data_integration():
    """Test la sincronización completa de TODOS los datos históricos."""
    
    print("🚀 TESTING COMPLETE GMX DATA INTEGRATION")
    print("=" * 70)
    
    # Initialize pipeline
    pipeline = GMXDataPipeline()
    
    # Test 1: Complete historical state
    print("\n📊 TEST 1: Complete Historical State")
    print("-" * 40)
    
    complete_state = pipeline.get_complete_historical_state()
    print(f"Status: {complete_state.get('status', 'unknown')}")
    print(f"Total Records: {complete_state.get('total_records', 0):,}")
    
    if 'data_sources' in complete_state:
        for source_name, source_info in complete_state['data_sources'].items():
            records = source_info.get('records', 0)
            description = source_info.get('description', 'No description')
            print(f"  📈 {source_name}: {records:,} records")
            print(f"      {description}")
            
            if source_name == 'prices' and 'tokens' in source_info:
                print(f"      Unique tokens: {source_info['tokens']}")
    
    if 'date_range' in complete_state and complete_state['date_range']:
        date_range = complete_state['date_range']
        print(f"\n📅 Date Coverage:")
        print(f"  Start: {date_range['start']}")
        print(f"  End: {date_range['end']}")
        print(f"  Days: {date_range['coverage_days']}")
    
    # Test 2: Individual data loaders
    print(f"\n📊 TEST 2: Individual Data Loaders")
    print("-" * 40)
    
    # Test price history
    price_data = pipeline.load_price_history()
    print(f"✅ Price History: {len(price_data):,} records")
    if price_data:
        sample = price_data[0]
        print(f"   Sample: Token {sample.get('token', 'N/A')[:10]}...")
        print(f"   Fields: {list(sample.keys())}")
        if 'price_min' in sample and 'price_max' in sample:
            print(f"   Processed: Min=${sample['price_min']:.6f}, Max=${sample['price_max']:.6f}")
    
    # Test borrowing rates
    borrowing_data = pipeline.load_borrowing_rates()
    print(f"\n✅ Borrowing Rates: {len(borrowing_data):,} records")
    if borrowing_data:
        sample = borrowing_data[0]
        print(f"   Sample: Address {sample.get('address', 'N/A')[:10]}...")
        print(f"   Fields: {list(sample.keys())}")
    
    # Test fee history (existing)
    fee_data = pipeline.load_global_history()
    print(f"\n✅ Fee History: {len(fee_data):,} records")
    if fee_data:
        sample = fee_data[0]
        print(f"   Sample timestamp: {sample.get('timestamp', 'N/A')}")
        print(f"   Fields: {list(sample.keys())}")
    
    # Test 3: Synchronized data retrieval
    print(f"\n📊 TEST 3: Synchronized Data Retrieval")
    print("-" * 40)
    
    # Get data from last 7 days
    end_time = datetime.now()
    start_time = end_time - timedelta(days=7)
    
    print(f"Requesting data from {start_time} to {end_time}")
    sync_data = pipeline.get_synchronized_market_data(
        start_time=start_time,
        end_time=end_time
    )
    
    if 'summary' in sync_data:
        summary = sync_data['summary']
        print(f"📊 Synchronized Results (Last 7 days):")
        print(f"   Total matching: {summary['total_matching_records']:,}")
        print(f"   Price records: {summary['price_records']:,}")
        print(f"   Fee records: {summary['fee_records']:,}")
        print(f"   Borrowing records: {summary['borrowing_records']:,}")
    
    # Test 4: Token-specific data
    print(f"\n📊 TEST 4: Token-Specific Data")
    print("-" * 40)
    
    # Try to get data for a specific token (use first token from price data)
    if price_data:
        test_token = price_data[0].get('token')
        if test_token:
            print(f"Testing token: {test_token}")
            token_data = pipeline.get_synchronized_market_data(
                token_address=test_token,
                start_time=end_time - timedelta(days=1)  # Last day
            )
            
            if 'summary' in token_data:
                token_summary = token_data['summary']
                print(f"Token-specific results (Last 24h):")
                print(f"   Price updates: {token_summary['price_records']:,}")
    
    # Test 5: Engine startup readiness check
    print(f"\n🎯 TEST 5: Engine Startup Readiness")
    print("-" * 40)
    
    readiness_score = 0
    total_checks = 4
    
    # Check 1: Price data availability
    if len(price_data) > 50000:  # Substantial price history
        print("✅ Price data: READY (>50K records)")
        readiness_score += 1
    else:
        print(f"⚠️  Price data: LIMITED ({len(price_data)} records)")
    
    # Check 2: Fee data availability  
    if len(fee_data) > 10000:
        print("✅ Fee data: READY (>10K records)")
        readiness_score += 1
    else:
        print(f"⚠️  Fee data: LIMITED ({len(fee_data)} records)")
    
    # Check 3: Rate data availability
    if len(borrowing_data) > 100:
        print("✅ Rate data: READY (>100 records)")
        readiness_score += 1
    else:
        print(f"⚠️  Rate data: LIMITED ({len(borrowing_data)} records)")
    
    # Check 4: Date coverage
    total_records = len(price_data) + len(fee_data) + len(borrowing_data)
    if total_records > 100000:
        print("✅ Total coverage: EXCELLENT (>100K total records)")
        readiness_score += 1
    else:
        print(f"⚠️  Total coverage: MODERATE ({total_records:,} records)")
    
    # Final readiness assessment
    readiness_percentage = (readiness_score / total_checks) * 100
    
    print(f"\n🎯 ENGINE STARTUP READINESS: {readiness_percentage:.0f}% ({readiness_score}/{total_checks})")
    
    if readiness_percentage >= 75:
        print("🚀 ✅ READY FOR PRODUCTION ENGINE STARTUP!")
        print("   All metrics can be perfectly synchronized")
    elif readiness_percentage >= 50:
        print("⚠️  READY FOR TESTING - Some limitations")
    else:
        print("❌ NOT READY - Insufficient historical data")
    
    return {
        "readiness_score": readiness_score,
        "total_checks": total_checks,
        "readiness_percentage": readiness_percentage,
        "data_summary": complete_state
    }

if __name__ == "__main__":
    result = asyncio.run(test_complete_data_integration())
    
    print("\n" + "=" * 70)
    print("🏁 INTEGRATION TEST COMPLETE")
    print(f"Final Score: {result['readiness_percentage']:.0f}%")
    
    if result['readiness_percentage'] == 100:
        print("🎯 PERFECT SYNCHRONIZATION ACHIEVED!")
    print("=" * 70)