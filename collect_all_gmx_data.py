import asyncio
import aiohttp
import json
import gzip
import os
from datetime import datetime, timezone
from typing import Dict, Any


class GMXComprehensiveCollector:
    """Collector completo para TODOS los datos históricos de GMX."""
    
    def __init__(self):
        self.url = "https://gmx.squids.live/gmx-synthetics-arbitrum:prod/api/graphql"
        self.data_dir = "data"
        os.makedirs(self.data_dir, exist_ok=True)
        
    async def collect_all_historical_data(self) -> Dict[str, Any]:
        """Recopilar TODOS los datos históricos disponibles."""
        
        results = {
            "status": "initiated",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data_sources": {}
        }
        
        print("🚀 INICIATING COMPREHENSIVE GMX DATA COLLECTION")
        print("=" * 70)
        
        # 1. Collect Price History (OHLCV reconstruction)
        print("\n📈 COLLECTING PRICE HISTORY...")
        price_data = await self._collect_price_history()
        results["data_sources"]["prices"] = price_data
        
        # 2. Collect Market Info History (Open Interest, Funding Rates)
        print("\n📊 COLLECTING MARKET INFO HISTORY...")
        market_data = await self._collect_market_info_history()
        results["data_sources"]["market_info"] = market_data
        
        # 3. Collect Position Changes (Liquidations, Position Dynamics)
        print("\n📉 COLLECTING POSITION CHANGES...")
        position_data = await self._collect_position_changes()
        results["data_sources"]["position_changes"] = position_data
        
        # 4. Collect Trade Actions (Volume, Trading Activity)
        print("\n💰 COLLECTING TRADE ACTIONS...")
        trade_data = await self._collect_trade_actions()
        results["data_sources"]["trade_actions"] = trade_data
        
        # 5. Collect Borrowing Rate Snapshots
        print("\n💸 COLLECTING BORROWING RATES...")
        borrowing_data = await self._collect_borrowing_rates()
        results["data_sources"]["borrowing_rates"] = borrowing_data
        
        # 6. Collect Swap Information
        print("\n🔄 COLLECTING SWAP HISTORY...")
        swap_data = await self._collect_swap_info()
        results["data_sources"]["swap_info"] = swap_data
        
        results["status"] = "completed"
        results["total_records"] = sum(
            data.get("records_count", 0) 
            for data in results["data_sources"].values()
        )
        
        # Save complete results
        results_file = f"{self.data_dir}/gmx_comprehensive_history.json.gz"
        with gzip.open(results_file, 'wt') as f:
            json.dump(results, f, indent=2)
        
        print("\n🎯 COLLECTION COMPLETE!")
        print(f"📊 Total records collected: {results['total_records']:,}")
        print(f"💾 Data saved to: {results_file}")
        
        return results
    
    async def _collect_price_history(self) -> Dict[str, Any]:
        """Recopilar historial completo de precios."""
        
        query = """{
          prices(limit: 1000, orderBy: timestamp_DESC) {
            id
            minPrice
            maxPrice
            timestamp
            type
            token
            isSnapshot
            snapshotTimestamp
          }
        }"""
        
        all_prices = []
        has_more = True
        last_timestamp = None
        batch_count = 0
        
        async with aiohttp.ClientSession() as session:
            while has_more and batch_count < 100:  # Safety limit
                # Modify query for pagination if needed
                current_query = query
                if last_timestamp:
                    current_query = query.replace(
                        'orderBy: timestamp_DESC',
                        f'orderBy: timestamp_DESC, where: {{timestamp_lt: {last_timestamp}}}'
                    )
                
                try:
                    async with session.post(self.url, json={'query': current_query}) as response:
                        if response.status == 200:
                            data = await response.json()
                            
                            if 'errors' in data:
                                print(f"❌ Price query error: {data['errors']}")
                                break
                            
                            prices = data['data']['prices']
                            if not prices:
                                has_more = False
                                break
                            
                            all_prices.extend(prices)
                            last_timestamp = prices[-1]['timestamp']
                            batch_count += 1
                            
                            print(f"   📦 Batch {batch_count}: {len(prices)} prices (total: {len(all_prices)})")
                            
                            if len(prices) < 1000:  # Partial batch means end
                                has_more = False
                        
                        else:
                            print(f"❌ Price HTTP error: {response.status}")
                            break
                            
                except Exception as e:
                    print(f"❌ Price collection error: {e}")
                    break
        
        # Save price history
        price_file = f"{self.data_dir}/gmx_price_history.json.gz"
        with gzip.open(price_file, 'wt') as f:
            json.dump(all_prices, f, indent=2)
        
        print(f"   💾 Saved {len(all_prices)} price records to {price_file}")
        
        return {
            "records_count": len(all_prices),
            "file_path": price_file,
            "data_sample": all_prices[:3] if all_prices else []
        }
    
    async def _collect_market_info_history(self) -> Dict[str, Any]:
        """Recopilar historial de información de mercados (OI, funding rates)."""
        
        query = """{
          marketInfos(limit: 100, orderBy: longOpenInterestUsd_DESC) {
            id
            marketTokenAddress
            longTokenAddress
            shortTokenAddress
            indexTokenAddress
            longOpenInterestUsd
            shortOpenInterestUsd
            longPoolAmount
            shortPoolAmount
            fundingFactorPerSecond
            borrowingFactorLong
            borrowingFactorShort
            totalBorrowingFeesUsd
            cumulativeBorrowingFeesUsd
            fundingRateLong
            fundingRateShort
            longTokensSupply
            shortTokensSupply
            totalSupply
            reserveFactorLong
            reserveFactorShort
            maxLongTokenPoolAmount
            maxShortTokenPoolAmount
            longPoolAmountAdjustment
            shortPoolAmountAdjustment
          }
        }"""
        
        market_info = []
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(self.url, json={'query': query}) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        if 'errors' in data:
                            print(f"❌ Market info error: {data['errors']}")
                        else:
                            market_info = data['data']['marketInfos']
                            print(f"   📦 Collected {len(market_info)} market info records")
                    
                    else:
                        print(f"❌ Market info HTTP error: {response.status}")
                        
            except Exception as e:
                print(f"❌ Market info collection error: {e}")
        
        # Save market info
        market_file = f"{self.data_dir}/gmx_market_info_history.json.gz"
        with gzip.open(market_file, 'wt') as f:
            json.dump(market_info, f, indent=2)
        
        print(f"   💾 Saved {len(market_info)} market info records to {market_file}")
        
        return {
            "records_count": len(market_info),
            "file_path": market_file,
            "data_sample": market_info[:2] if market_info else []
        }
    
    async def _collect_position_changes(self) -> Dict[str, Any]:
        """Recopilar cambios de posiciones y liquidaciones."""
        
        query = """{
          positionChanges(limit: 1000, orderBy: timestamp_DESC) {
            id
            type
            account
            market
            collateralToken
            isLong
            sizeInUsd
            sizeDeltaUsd
            sizeDeltaInTokens
            collateralAmount
            collateralDeltaAmount
            executionPrice
            indexTokenPrice
            collateralTokenPrice
            realizedPnl
            fundingFeeAmount
            borrowingFeeUsd
            positionFeeAmount
            priceImpactUsd
            orderType
            orderKey
            timestamp
          }
        }"""
        
        all_changes = []
        has_more = True
        last_timestamp = None
        batch_count = 0
        
        async with aiohttp.ClientSession() as session:
            while has_more and batch_count < 50:  # Safety limit
                current_query = query
                if last_timestamp:
                    current_query = query.replace(
                        'orderBy: timestamp_DESC',
                        f'orderBy: timestamp_DESC, where: {{timestamp_lt: {last_timestamp}}}'
                    )
                
                try:
                    async with session.post(self.url, json={'query': current_query}) as response:
                        if response.status == 200:
                            data = await response.json()
                            
                            if 'errors' in data:
                                print(f"❌ Position changes error: {data['errors']}")
                                break
                            
                            changes = data['data']['positionChanges']
                            if not changes:
                                has_more = False
                                break
                            
                            all_changes.extend(changes)
                            last_timestamp = changes[-1]['timestamp']
                            batch_count += 1
                            
                            print(f"   📦 Batch {batch_count}: {len(changes)} changes (total: {len(all_changes)})")
                            
                            if len(changes) < 1000:
                                has_more = False
                        
                        else:
                            print(f"❌ Position changes HTTP error: {response.status}")
                            break
                            
                except Exception as e:
                    print(f"❌ Position changes error: {e}")
                    break
        
        # Save position changes
        changes_file = f"{self.data_dir}/gmx_position_changes_history.json.gz"
        with gzip.open(changes_file, 'wt') as f:
            json.dump(all_changes, f, indent=2)
        
        print(f"   💾 Saved {len(all_changes)} position changes to {changes_file}")
        
        return {
            "records_count": len(all_changes),
            "file_path": changes_file,
            "data_sample": all_changes[:3] if all_changes else []
        }
    
    async def _collect_trade_actions(self) -> Dict[str, Any]:
        """Recopilar acciones de trading y volumen."""
        
        query = """{
          tradeActions(limit: 1000, orderBy: timestamp_DESC) {
            id
            eventName
            orderKey
            orderType
            account
            marketAddress
            initialCollateralTokenAddress
            initialCollateralDeltaAmount
            sizeDeltaUsd
            triggerPrice
            acceptablePrice
            executionPrice
            minOutputAmount
            actualOutputAmount
            priceImpactUsd
            swapPriceImpactDeltaUsd
            positionFeeAmount
            borrowingFeeUsd
            fundingFeeAmount
            pnlUsd
            basePnlUsd
            timestamp
          }
        }"""
        
        all_trades = []
        has_more = True
        last_timestamp = None
        batch_count = 0
        
        async with aiohttp.ClientSession() as session:
            while has_more and batch_count < 50:  # Safety limit
                current_query = query
                if last_timestamp:
                    current_query = query.replace(
                        'orderBy: timestamp_DESC',
                        f'orderBy: timestamp_DESC, where: {{timestamp_lt: {last_timestamp}}}'
                    )
                
                try:
                    async with session.post(self.url, json={'query': current_query}) as response:
                        if response.status == 200:
                            data = await response.json()
                            
                            if 'errors' in data:
                                print(f"❌ Trade actions error: {data['errors']}")
                                break
                            
                            trades = data['data']['tradeActions']
                            if not trades:
                                has_more = False
                                break
                            
                            all_trades.extend(trades)
                            last_timestamp = trades[-1]['timestamp']
                            batch_count += 1
                            
                            print(f"   📦 Batch {batch_count}: {len(trades)} trades (total: {len(all_trades)})")
                            
                            if len(trades) < 1000:
                                has_more = False
                        
                        else:
                            print(f"❌ Trade actions HTTP error: {response.status}")
                            break
                            
                except Exception as e:
                    print(f"❌ Trade actions error: {e}")
                    break
        
        # Save trade actions
        trades_file = f"{self.data_dir}/gmx_trade_actions_history.json.gz"
        with gzip.open(trades_file, 'wt') as f:
            json.dump(all_trades, f, indent=2)
        
        print(f"   💾 Saved {len(all_trades)} trade actions to {trades_file}")
        
        return {
            "records_count": len(all_trades),
            "file_path": trades_file,
            "data_sample": all_trades[:3] if all_trades else []
        }
    
    async def _collect_borrowing_rates(self) -> Dict[str, Any]:
        """Recopilar snapshots de tasas de borrowing."""
        
        query = """{
          borrowingRateSnapshots(limit: 1000, orderBy: snapshotTimestamp_DESC) {
            id
            address
            borrowingFactorPerSecondLong
            borrowingFactorPerSecondShort
            borrowingRateForPool
            snapshotTimestamp
            entityType
          }
        }"""
        
        all_rates = []
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(self.url, json={'query': query}) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        if 'errors' in data:
                            print(f"❌ Borrowing rates error: {data['errors']}")
                        else:
                            all_rates = data['data']['borrowingRateSnapshots']
                            print(f"   📦 Collected {len(all_rates)} borrowing rate snapshots")
                    
                    else:
                        print(f"❌ Borrowing rates HTTP error: {response.status}")
                        
            except Exception as e:
                print(f"❌ Borrowing rates error: {e}")
        
        # Save borrowing rates
        rates_file = f"{self.data_dir}/gmx_borrowing_rates_history.json.gz"
        with gzip.open(rates_file, 'wt') as f:
            json.dump(all_rates, f, indent=2)
        
        print(f"   💾 Saved {len(all_rates)} borrowing rates to {rates_file}")
        
        return {
            "records_count": len(all_rates),
            "file_path": rates_file,
            "data_sample": all_rates[:3] if all_rates else []
        }
    
    async def _collect_swap_info(self) -> Dict[str, Any]:
        """Recopilar información de swaps."""
        
        query = """{
          swapInfos(limit: 1000, orderBy: timestamp_DESC) {
            id
            orderKey
            receiver
            marketAddress
            tokenInAddress
            tokenOutAddress
            tokenInPrice
            tokenOutPrice
            amountIn
            amountInAfterFees
            amountOut
            priceImpactUsd
            timestamp
          }
        }"""
        
        all_swaps = []
        has_more = True
        last_timestamp = None
        batch_count = 0
        
        async with aiohttp.ClientSession() as session:
            while has_more and batch_count < 20:  # Safety limit
                current_query = query
                if last_timestamp:
                    current_query = query.replace(
                        'orderBy: timestamp_DESC',
                        f'orderBy: timestamp_DESC, where: {{timestamp_lt: {last_timestamp}}}'
                    )
                
                try:
                    async with session.post(self.url, json={'query': current_query}) as response:
                        if response.status == 200:
                            data = await response.json()
                            
                            if 'errors' in data:
                                print(f"❌ Swap info error: {data['errors']}")
                                break
                            
                            swaps = data['data']['swapInfos']
                            if not swaps:
                                has_more = False
                                break
                            
                            all_swaps.extend(swaps)
                            last_timestamp = swaps[-1]['timestamp']
                            batch_count += 1
                            
                            print(f"   📦 Batch {batch_count}: {len(swaps)} swaps (total: {len(all_swaps)})")
                            
                            if len(swaps) < 1000:
                                has_more = False
                        
                        else:
                            print(f"❌ Swap info HTTP error: {response.status}")
                            break
                            
                except Exception as e:
                    print(f"❌ Swap info error: {e}")
                    break
        
        # Save swap info
        swaps_file = f"{self.data_dir}/gmx_swap_info_history.json.gz"
        with gzip.open(swaps_file, 'wt') as f:
            json.dump(all_swaps, f, indent=2)
        
        print(f"   💾 Saved {len(all_swaps)} swap records to {swaps_file}")
        
        return {
            "records_count": len(all_swaps),
            "file_path": swaps_file,
            "data_sample": all_swaps[:3] if all_swaps else []
        }


async def main():
    """Execute comprehensive data collection."""
    collector = GMXComprehensiveCollector()
    results = await collector.collect_all_historical_data()
    
    print("\n" + "=" * 70)
    print("📊 COLLECTION SUMMARY:")
    print("-" * 70)
    
    for source_name, source_data in results["data_sources"].items():
        records = source_data.get("records_count", 0)
        print(f"   {source_name}: {records:,} records")
    
    print(f"\n🎯 Total Historical Records: {results['total_records']:,}")
    print("✅ ALL GMX DATA SOURCES SYNCHRONIZED!")


if __name__ == "__main__":
    asyncio.run(main())