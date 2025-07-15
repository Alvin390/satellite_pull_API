// src/evalscript_wbm.js

// NDWI-based water body mask: NDWI = (Green - NIR) / (Green + NIR)
// Sentinel-2 B03 (Green), B08 (NIR)
return {
  input: ["B03", "B08", "dataMask"],
  output: {
    bands: 1,
    sampleType: "UINT8"
  },
  evaluatePixel: function(sample) {
    if (sample.dataMask === 0) return [0];

    let ndwi = (sample.B03 - sample.B08) / (sample.B03 + sample.B08);
    return [ndwi > 0.3 ? 1 : 0];  // Water if NDWI > 0.3
  }
};
