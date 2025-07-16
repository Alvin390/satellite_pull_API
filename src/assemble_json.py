import json
import os
import numpy as np
import datetime as dt
from .fetch_satellite import summary, results
from .fetch_climate import precip, hourly

def convert_types(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    return str(obj)

payload = dict(
    timestamp=dt.datetime.utcnow().isoformat() + "Z",
    satellite_water_data=dict(summary=summary, water_bodies=results),
    climate=dict(
        precipitation_imerg_mm_per_hr=precip if precip is not None else "unavailable",
        open_meteo=hourly if hourly is not None else "unavailable"
    )
)

os.makedirs('output', exist_ok=True)
with open('output/latest_water_kenya.json', 'w') as f:
    json.dump(payload, f, indent=2, default=convert_types)
print("✅  Wrote output/latest_water_kenya.json")
