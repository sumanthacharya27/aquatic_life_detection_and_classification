import h5py
import json

model_path = "model/aquatic_model_v1.h5"

with h5py.File(model_path, "r+") as f:
    model_config = f.attrs.get("model_config")
    if isinstance(model_config, bytes):
        model_config = model_config.decode("utf-8")

    config = json.loads(model_config)

    def fix_layer(layer):
        cfg = layer.get("config", {})
        
        # Fix InputLayer issues
        cfg.pop("optional", None)
        if "batch_shape" in cfg:
            cfg["batch_input_shape"] = cfg.pop("batch_shape")
        
        # Fix DTypePolicy — replace object with plain string
        if "dtype" in cfg and isinstance(cfg["dtype"], dict):
            dtype_val = cfg["dtype"].get("config", {}).get("name", "float32")
            cfg["dtype"] = dtype_val

        # Fix initializers — remove 'module' key (Keras 3 format)
        for key in ["kernel_initializer", "bias_initializer",
                    "gamma_initializer", "beta_initializer",
                    "moving_mean_initializer", "moving_variance_initializer"]:
            if key in cfg and isinstance(cfg[key], dict):
                cfg[key].pop("module", None)
                cfg[key].pop("registered_name", None)

        # Recurse into sublayers
        for sublayer in cfg.get("layers", []):
            fix_layer(sublayer)

    for layer in config.get("config", {}).get("layers", []):
        fix_layer(layer)

    f.attrs["model_config"] = json.dumps(config).encode("utf-8")
    print("✅ Model patched successfully!")