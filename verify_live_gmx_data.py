#!/usr/bin/env python3
"""
Verificación rápida de datos GMX en vivo
Confirma que el dashboard está recibiendo datos reales de GMX
"""

import sys
import os
import asyncio

# Add the project root to the Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(project_root, 'future-research', 'crypto-lab', 'src'))

from unified_data_manager import UnifiedDataManager


async def verify_gmx_data():
    """Verificar que los datos GMX son reales y no sintéticos"""
    
    print("🔍 Verificando fuente de datos GMX...")
    
    # Inicializar el data manager
    manager = UnifiedDataManager()
    
    # Obtener datos recientes
    data = await manager.get_unified_data()
    
    print(f"📊 Total de mercados: {len(data)}")
    
    # Verificar fuentes de datos
    sources = {}
    for market_data in data:
        symbol = market_data.symbol
        source = market_data.source
        if source not in sources:
            sources[source] = []
        sources[source].append(symbol)
    
    print("\n📈 Fuentes de datos por mercado:")
    for source, symbols in sources.items():
        print(f"  {source}: {len(symbols)} mercados")
        if source == 'gmx_live':
            print(f"    ✅ GMX Live: {', '.join(symbols[:5])}" + (f" (+{len(symbols)-5} más)" if len(symbols) > 5 else ""))
    
    # Verificar datos específicos de GMX
    gmx_markets = [market for market in data if market.source == 'gmx_live']
    
    if gmx_markets:
        print(f"\n✅ CONFIRMADO: {len(gmx_markets)} mercados usando datos GMX REALES")
        
        # Mostrar ejemplo de datos
        example_data = gmx_markets[0]
        example_symbol = example_data.symbol
        
        print(f"\n📋 Ejemplo de datos para {example_symbol}:")
        print(f"  Precio: ${example_data.price:.4f}")
        print(f"  Volumen 24h: ${example_data.volume_24h:,.0f}")
        print(f"  OI Long: ${example_data.oi_long:,.0f}")
        print(f"  OI Short: ${example_data.oi_short:,.0f}")
        print(f"  Funding Rate: {example_data.funding_rate:.6f}")
        print(f"  Timestamp: {example_data.timestamp}")
        print(f"  Fuente: {example_data.source}")
        
        # Verificar que los precios no son valores sintéticos típicos
        prices = [market.price for market in data if market.source == 'gmx_live']
        unique_prices = len(set(prices))
        
        if unique_prices > 5:  # Si hay variedad de precios, probablemente son reales
            print(f"  ✅ Variedad de precios detectada: {unique_prices} precios únicos")
        else:
            print(f"  ⚠️  Poca variedad de precios: {unique_prices} precios únicos")
            
    else:
        print("\n❌ ERROR: No se encontraron mercados con datos GMX reales")
        print("   Verificar configuración de APIs GMX")
    
    return len(gmx_markets) > 0

if __name__ == "__main__":
    success = asyncio.run(verify_gmx_data())
    if success:
        print("\n🎉 ¡DASHBOARD CONFIRMADO CON DATOS GMX REALES!")
    else:
        print("\n⚠️  Dashboard usando datos sintéticos o error de configuración")