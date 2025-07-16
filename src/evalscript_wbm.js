function setup() {
    return {
        input: ["B02", "B03", "B04", "B05", "B08", "dataMask"],
        output: [
            { id: "water_mask", bands: 1, sampleType: "UINT8" },
            { id: "turbidity", bands: 1, sampleType: "FLOAT32" },
            { id: "chlorophyll", bands: 1, sampleType: "FLOAT32" }
        ]
    };
}

function evaluatePixel(sample) {
    if (sample.dataMask === 0) {
        return {
            water_mask: [0],
            turbidity: [0],
            chlorophyll: [0]
        };
    }

    // NDWI for water detection: (Green - NIR) / (Green + NIR)
    let ndwi = (sample.B03 - sample.B08) / (sample.B03 + sample.B08);
    
    // NDTI for turbidity: (Red - Green) / (Red + Green)
    let ndti = (sample.B04 - sample.B03) / (sample.B04 + sample.B03);
    
    // Chlorophyll-a proxy: Blue / Red-edge (simplified for inland waters)
    let chl = sample.B02 / (sample.B05 + 0.0001); // Avoid division by zero

    return {
        water_mask: [ndwi > 0.3 ? 1 : 0],
        turbidity: [ndti], // Unitless, higher values indicate more turbidity
        chlorophyll: [chl] // Proxy in mg/m³, approximate
    };
}
