import sys
from pathlib import Path
sys.path.insert(0,str(Path(sys.argv[1]).resolve()))
import gc,json,statistics,time,tracemalloc
import networkx as nx
import numpy as np
from tnfr.physics.structural_diffusion import compute_diffusion_energy,current_divergence
records=[]
for size in (500,2000):
    graph=nx.path_graph(size)
    rng=np.random.default_rng(7)
    for node,epi,frequency in zip(graph,rng.normal(size=size),rng.uniform(.1,2,size)):
        graph.nodes[node].update(EPI=float(epi),nu_f=float(frequency),theta=0.)
    for reader in (current_divergence,compute_diffusion_energy):
        reader(graph)
        elapsed=[]
        for repeat in range(5):
            start=time.perf_counter()
            result=reader(graph)
            elapsed.append(time.perf_counter()-start)
        gc.collect()
        tracemalloc.start()
        reader(graph)
        _,peak=tracemalloc.get_traced_memory()
        tracemalloc.stop()
        vector=result[1] if reader is current_divergence else result.gradient
        record=dict(nodes=size,reader=reader.__name__,median_seconds=statistics.median(elapsed),peak_bytes=peak,gradient_norm=float(np.linalg.norm(vector)))
        if reader is compute_diffusion_energy:
            record.update(energy=result.energy,energy_rate=result.energy_rate)
        records.append(record)
print(json.dumps(records,indent=2))
