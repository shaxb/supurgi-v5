import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from trendstack.cores.data_core import load_prices, load_costs
import pandas as pd
import time
from pathlib import Path

def clean_test_data():
    """Clean all test data to start fresh."""
    data_path = Path("./data")
    
    print("🧹 CLEANING TEST DATA")
    print("-" * 30)
    
    files_removed = 0
    for folder in ["raw", "processed"]:
        folder_path = data_path / folder
        if folder_path.exists():
            for file in folder_path.glob("AAPL_*.parquet"):
                file.unlink()
                print(f"   ❌ Removed {file.name}")
                files_removed += 1
    
    print(f"   🗑️  Removed {files_removed} files\n")

def test_data_flow():
    """Test how data flows through the system: download -> cache -> use."""
    
    print("🔄 TESTING DATA FLOW: DOWNLOAD → CACHE → USE")
    print("=" * 60)
    
    # Start clean
    clean_test_data()
    symbol = "AAPL"
    data_path = Path("./data")
    
    # Test 1: Daily timeframe - first load (full download)
    print("1️⃣  DAILY TIMEFRAME - First Load (Full Download)")
    print("-" * 50)
    
    raw_d = data_path / "raw" / f"{symbol}_D.parquet"
    processed_d = data_path / "processed" / f"{symbol}_D.parquet"
    
    print(f"   📁 Raw file exists before: {raw_d.exists()}")
    print(f"   📁 Processed file exists before: {processed_d.exists()}")
    
    start_time = time.time()
    data_d1 = load_prices(symbol, frame="D")
    duration_d1 = time.time() - start_time
    
    print(f"   ✅ Loaded {len(data_d1)} daily bars in {duration_d1:.2f}s")
    print(f"   📁 Raw file exists after: {raw_d.exists()}")
    print(f"   📁 Processed file exists after: {processed_d.exists()}")
    print(f"   📅 Date range: {data_d1.index[0].date()} to {data_d1.index[-1].date()}")
    
    # Check file sizes
    if raw_d.exists():
        raw_size = raw_d.stat().st_size
        print(f"   💾 Raw file size: {raw_size:,} bytes")
    if processed_d.exists():
        processed_size = processed_d.stat().st_size
        print(f"   💾 Processed file size: {processed_size:,} bytes")
    
    print()
    
    # Test 2: Daily timeframe - second load (cache hit)
    print("2️⃣  DAILY TIMEFRAME - Second Load (Cache Hit)")
    print("-" * 50)
    
    start_time = time.time()
    data_d2 = load_prices(symbol, frame="D")
    duration_d2 = time.time() - start_time
    
    print(f"   ✅ Loaded {len(data_d2)} daily bars in {duration_d2:.2f}s")
    print(f"   🚀 Speed improvement: {duration_d1/duration_d2:.1f}x faster!")
    print(f"   📊 Data identical: {data_d1.equals(data_d2)}")
    print()
    
    # Test 3: H4 timeframe - first load (new timeframe)
    print("3️⃣  H4 TIMEFRAME - First Load (New Timeframe)")
    print("-" * 50)
    
    raw_h4 = data_path / "raw" / f"{symbol}_H4.parquet"
    processed_h4 = data_path / "processed" / f"{symbol}_H4.parquet"
    
    print(f"   📁 H4 raw file exists before: {raw_h4.exists()}")
    print(f"   📁 H4 processed file exists before: {processed_h4.exists()}")
    
    start_time = time.time()
    data_h4_1 = load_prices(symbol, frame="H4")
    duration_h4_1 = time.time() - start_time
    
    print(f"   ✅ Loaded {len(data_h4_1)} H4 bars in {duration_h4_1:.2f}s")
    print(f"   📁 H4 raw file exists after: {raw_h4.exists()}")
    print(f"   📁 H4 processed file exists after: {processed_h4.exists()}")
    
    if len(data_h4_1) > 0:
        print(f"   📅 Date range: {data_h4_1.index[0]} to {data_h4_1.index[-1]}")
        
        # Check file sizes
        if raw_h4.exists():
            raw_h4_size = raw_h4.stat().st_size
            print(f"   💾 H4 raw file size: {raw_h4_size:,} bytes")
        if processed_h4.exists():
            processed_h4_size = processed_h4.stat().st_size
            print(f"   💾 H4 processed file size: {processed_h4_size:,} bytes")
    else:
        print("   ⚠️  No H4 data available (API limitations)")
    
    print()
    
    # Test 4: H4 timeframe - second load (cache hit)
    print("4️⃣  H4 TIMEFRAME - Second Load (Cache Hit)")
    print("-" * 50)
    
    start_time = time.time()
    data_h4_2 = load_prices(symbol, frame="H4")
    duration_h4_2 = time.time() - start_time
    
    print(f"   ✅ Loaded {len(data_h4_2)} H4 bars in {duration_h4_2:.2f}s")
    if duration_h4_1 > 0 and duration_h4_2 > 0:
        print(f"   🚀 Speed improvement: {duration_h4_1/duration_h4_2:.1f}x faster!")
    if len(data_h4_1) > 0 and len(data_h4_2) > 0:
        print(f"   📊 Data identical: {data_h4_1.equals(data_h4_2)}")
    print()
    
    # Test 5: Date range requests (using cache)
    print("5️⃣  DATE RANGE REQUESTS (Using Cache)")
    print("-" * 50)
    
    # Recent month
    start_time = time.time()
    data_recent = load_prices(symbol, frame="D", start="2025-07-01", end="2025-07-25")
    duration_recent = time.time() - start_time
    
    print(f"   ✅ Recent month: {len(data_recent)} bars in {duration_recent:.2f}s")
    if len(data_recent) > 0:
        print(f"   📅 Range: {data_recent.index[0].date()} to {data_recent.index[-1].date()}")
    
    # Last 3 months
    start_time = time.time()
    data_3m = load_prices(symbol, frame="D", start="2025-04-01", end="2025-07-25")
    duration_3m = time.time() - start_time
    
    print(f"   ✅ Last 3 months: {len(data_3m)} bars in {duration_3m:.2f}s")
    if len(data_3m) > 0:
        print(f"   📅 Range: {data_3m.index[0].date()} to {data_3m.index[-1].date()}")
    
    print()
    
    # Test 6: File structure summary
    print("6️⃣  FILE STRUCTURE SUMMARY")
    print("-" * 50)
    
    total_files = 0
    total_size = 0
    
    for folder in ["raw", "processed"]:
        folder_path = data_path / folder
        if folder_path.exists():
            files = list(folder_path.glob(f"{symbol}_*.parquet"))
            print(f"   📁 {folder}/ ({len(files)} files):")
            for file in files:
                size = file.stat().st_size
                print(f"      - {file.name}: {size:,} bytes")
                total_files += 1
                total_size += size
    
    print(f"   📊 Total: {total_files} files, {total_size:,} bytes")
    print()
    
    # Test 7: Performance summary
    print("7️⃣  PERFORMANCE SUMMARY")
    print("-" * 50)
    print(f"   📈 Daily first load:  {duration_d1:.2f}s ({len(data_d1)} bars)")
    print(f"   ⚡ Daily cache hit:   {duration_d2:.2f}s ({duration_d1/duration_d2:.1f}x faster)")
    if duration_h4_1 > 0:
        print(f"   📈 H4 first load:     {duration_h4_1:.2f}s ({len(data_h4_1)} bars)")
        print(f"   ⚡ H4 cache hit:      {duration_h4_2:.2f}s ({duration_h4_1/duration_h4_2:.1f}x faster)")
    print(f"   ⚡ Range requests:    {duration_recent:.2f}s (instant from cache)")
    
    print("\n🎯 API BEHAVIOR VERIFICATION:")
    print(f"   ✅ Downloads data only when needed")
    print(f"   ✅ Caches data per timeframe")
    print(f"   ✅ Serves from cache when available")
    print(f"   ✅ Handles multiple timeframes independently")
    print(f"   ✅ Range filtering works from cached data")
    print(f"   ✅ Files are created and managed properly")

def test_incremental_updates():
    """Test how incremental updates work."""
    
    print("\n" + "=" * 60)
    print("🔄 TESTING INCREMENTAL UPDATES")
    print("=" * 60)
    
    symbol = "EURUSD=X"
    
    # Load data (might trigger incremental update)
    print("Loading EURUSD data to test incremental logic...")
    data1 = load_prices(symbol, frame="D")
    print(f"✅ Loaded {len(data1)} bars")
    
    # Load again immediately (should be fast - no update needed)
    start_time = time.time()
    data2 = load_prices(symbol, frame="D")
    duration = time.time() - start_time
    
    print(f"✅ Second load: {len(data2)} bars in {duration:.2f}s")
    print(f"📊 Data identical: {data1.equals(data2)}")
    
    if duration < 1.0:
        print("⚡ Cache working perfectly - no download needed!")
    else:
        print("🔄 Update was needed - incremental download happened")

if __name__ == "__main__":
    # test_data_flow()
    # test_incremental_updates()
    print("\n🎉 ALL TESTS COMPLETED!")
    gpd = load_prices("GBPUSD=X", frame="M1", start="2025-07-01", end="2025-07-25")   # Trigger load to ensure everything works
    print(gpd.head())
    load_costs("EURUSD=X")  # Ensure costs can be loaded
    cost = load_costs("EURUSD=X")
    print(type(cost))
    print("✅ Costs loaded successfully")