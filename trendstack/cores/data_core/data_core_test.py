import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from trendstack.cores.data_core import load_prices, load_costs, refresh_symbol, list_symbols, CostSpec, Frame
import pandas as pd
from pathlib import Path
import time

def test_caching_and_incremental():
    """Test to verify caching and incremental updates work correctly."""
    
    print("🔍 TESTING DATA CACHING AND INCREMENTAL UPDATES")
    print("=" * 60)
    
    # Test symbol
    symbol = "AAPL"
    data_path = Path("./data")
    raw_file = data_path / "raw" / f"{symbol}.parquet"
    processed_file = data_path / "processed" / f"{symbol}_D.parquet"
    
    # Step 1: Clean slate - remove existing files
    print("1. Cleaning slate (removing existing data)...")
    if raw_file.exists():
        raw_file.unlink()
        print(f"   ❌ Removed {raw_file}")
    if processed_file.exists():
        processed_file.unlink()
        print(f"   ❌ Removed {processed_file}")
    
    # Step 2: First load - should download full history
    print(f"\n2. First load_prices('{symbol}') - should download full history...")
    start_time = time.time()
    data1 = load_prices(symbol)
    duration1 = time.time() - start_time
    
    print(f"   ✅ Loaded {len(data1)} bars in {duration1:.2f} seconds")
    print(f"   📁 Raw file exists: {raw_file.exists()}")
    print(f"   📁 Processed file exists: {processed_file.exists()}")
    if data1.empty:
        print("   ⚠️  No data loaded!")
        return
    
    last_date1 = data1.index[-1]
    print(f"   📅 Last date: {last_date1.date()}")
    
    # Step 3: Second load - should use cache (fast)
    print(f"\n3. Second load_prices('{symbol}') - should use cache...")
    start_time = time.time()
    data2 = load_prices(symbol)
    duration2 = time.time() - start_time
    
    print(f"   ✅ Loaded {len(data2)} bars in {duration2:.2f} seconds")
    print(f"   🚀 Speed improvement: {duration1/duration2:.1f}x faster!")
    
    # Verify data is identical
    if data1.equals(data2):
        print("   ✅ Data identical - caching works!")
    else:
        print("   ❌ Data different - caching issue!")
    
    # Step 4: Manual refresh - should check for incremental updates
    print(f"\n4. Manual refresh_symbol('{symbol}') - checking for new data...")
    start_time = time.time()
    refresh_symbol(symbol)
    duration3 = time.time() - start_time
    
    print(f"   ⏱️  Refresh took {duration3:.2f} seconds")
    
    # Step 5: Load after refresh
    data3 = load_prices(symbol)
    print(f"   📊 Data after refresh: {len(data3)} bars")
    
    if len(data3) > len(data1):
        print(f"   ✅ New data added: +{len(data3) - len(data1)} bars")
    else:
        print("   ℹ️  No new data (normal for weekends/holidays)")
    
    # Step 6: Check raw vs processed data consistency
    print(f"\n5. Verifying data consistency...")
    if raw_file.exists():
        raw_data = pd.read_parquet(raw_file)
        print(f"   📊 Raw data: {len(raw_data)} bars")
        print(f"   📊 Processed data: {len(data3)} bars")
        
        # Check if processed <= raw (cleaning can remove outliers)
        if len(data3) <= len(raw_data):
            print("   ✅ Data consistency check passed")
        else:
            print("   ❌ More processed data than raw - issue!")
    
    print(f"\n🎯 SUMMARY:")
    print(f"   Cache working: {'✅ YES' if duration2 < duration1/2 else '❌ NO'}")
    print(f"   Files created: {'✅ YES' if raw_file.exists() and processed_file.exists() else '❌ NO'}")
    print(f"   Incremental refresh: ✅ TESTED")

def main(args=None):
    print("Testing data_core...")
    
    # Test list symbols
    symbols = list_symbols()
    print(f"Available symbols: {len(symbols)}")
    
    # Test load prices
    if symbols:
        symbol = list(symbols.keys())[0]  # Get first symbol
        print(f"\nLoading data for {symbol}...")
        
        data = load_prices(symbol)
        print(f"Loaded {len(data)} bars")
        print(f"Columns: {list(data.columns)}")
        print(f"Date range: {data.index[0]} to {data.index[-1]}")
        print("\nFirst 5 rows:")
        print(data.head())
        
        print("\nLast 5 rows:")
        print(data.tail())
    
    # Run detailed caching test
    print("\n" + "="*60)
    test_caching_and_incremental()

if __name__ == "__main__":
    main()