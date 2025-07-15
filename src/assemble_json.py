import json
import datetime as dt
from fetch_satellite import summary, results
from fetch_climate import precip, hourly

payload = dict(
    timestamp=dt.datetime.utcnow().isoformat() + "Z",
    satellite_water_data=dict(summary=summary, water_bodies=results),
    climate=dict(
        precipitation_imerg_mm_per_hr=precip,
        open_meteo=hourly
    )
)

with open('output/latest_water_kenya.json', 'w') as f:
    json.dump(payload, f, indent=2)
print("✅  Wrote output/latest_water_kenya.json")
