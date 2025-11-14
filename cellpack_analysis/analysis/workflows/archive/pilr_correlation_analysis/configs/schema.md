```json
{
    // Workflow name for the analysis
    "workflow": "peroxisome",

    // Experiment type. This is the subfolder in the `workflow` folder
    "experiment": "rules_shape",

    // Rules defining PILRs
    "rules": {
        // Rule for observed peroxisomes
        "SLC25A17": {
            "structure_id": "SLC25A17", // Identifier for the structure used to create simulated data
            "label": "Observed peroxisomes", // Label for visualization
            "color": "#FF5733", // Hex color code for representation
            "dsphere": true, // Indicates whether cell_ids are from dsphere
        },
        "random": {
            "structure_id": "SLC25A17",
            "label": "Random", 
            "color": "#C0C0C0" 
        },
        "nucleus_gradient_strong": {
            "structure_id": "SLC25A17", 
            "label": "Nucleus", 
            "color": "#0000FF" 
        },
        "membrane_gradient_strong": {
            "structure_id": "SLC25A17", 
            "label": "Membrane", 
            "color": "#008000"
        }
    }
}
```