function setup() {
    return {
        input: ["B02", "B03", "B04", "B05", "B08", "B11", "SCL", "dataMask"],
        output: [
            { id: "water_mask", bands: 1, sampleType: "UINT8" },
            { id: "turbidity", bands: 1, sampleType: "FLOAT32" },
            { id: "chlorophyll", bands: 1, sampleType: "FLOAT32" }
        ]
    };
}

function evaluatePixel(sample) {
    // Mask clouds, cloud shadows, and invalid pixels
    if (sample.dataMask === 0 || sample.SCL === 3 || sample.SCL === 8 || sample.SCL === 9) {
        return {
            water_mask: [0],
            turbidity: [0],
            chlorophyll: [0]
        };
    }

    // NDWI for water detection: (Green - NIR) / (Green + NIR)
    let ndwi = (sample.B03 - sample.B08) / (sample.B03 + sample.B08 + 0.0001);
    // Confirm water with SWIR (B11) to reduce false positives
    let isWater = ndwi > 0.2 && sample.B11 < 0.1;

    // NDTI for turbidity: (Red - Green) / (Red + Green)
    let ndti = (sample.B04 - sample.B03) / (sample.B04 + sample.B03 + 0.0001);
    // Clamp NDTI to [0, 1] for meaningful turbidity values
    let clampedNdti = Math.max(0, Math.min(ndti, 1));

    // Simplified OC3-like chlorophyll-a proxy for inland waters
    // Uses log-transformed ratio of blue (B02) to red (B04) and NIR (B08)
    let chlRatio = Math.max(sample.B02, sample.B04) / (sample.B08 + 0.0001);
    let chl = 10 * Math.log10(chlRatio + 0.0001); // Approximate mg/m³
    // Clamp chlorophyll to [0, 100] mg/m³
    let clampedChl = Math.max(0, Math.min(chl, 100));

    return {
        water_mask: [isWater ? 1 : 0],
        turbidity: [clampedNdti],
        chlorophyll: [clampedChl]
    };
}