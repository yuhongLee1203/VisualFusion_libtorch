import os
import torch
import tensorrt as trt
from model_jit.SemLA import SemLA

device = torch.device("cuda")

matcher = SemLA(device, torch.float16)
matcher.load_state_dict(torch.load(f"./reg.ckpt"), strict=False)

matcher = matcher.eval().to(device, dtype=torch.float16)

torch_input_1 = torch.randn(1, 1, 256, 320).half().to(device)
torch_input_2 = torch.randn(1, 1, 256, 320).half().to(device)

torch.onnx.export(
    matcher,
    (torch_input_1, torch_input_2),
    "./SemLA_onnx_tensorrt.onnx",
    verbose=True,
    opset_version=12,
    input_names=["vi_img", "ir_img"],
    output_names=["mkpt0", "mkpt1", "feat_sa_vi", "feat_sa_ir", "sa_ir"],
    dynamic_axes={
        "vi_img": [2, 3],
        "ir_img": [2, 3],
        "mkpt0": [0],
        "mkpt1": [0],
        "feat_sa_vi": [2, 3],
        "feat_sa_ir": [2, 3],
        "sa_ir": [2, 3],
    },
)

# ----- Tensorrt -----

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)


def printShape(engine):
    for i in range(engine.num_bindings):
        if engine.binding_is_input(i):
            print(
                "input layer: {}, shape is: {} ".format(i, engine.get_binding_shape(i))
            )
        else:
            print(
                "output layer: {} shape is: {} ".format(i, engine.get_binding_shape(i))
            )


def onnx2trt(onnx_path, engine_path):
    # Create a builder
    builder = trt.Builder(TRT_LOGGER)

    # Specify that the network should be created with an explicit batch dimension
    builder.network_creation_flags |= EXPLICIT_BATCH

    # Create a network
    network = builder.create_network()

    # add layers to the network
    with trt.OnnxParser(network, TRT_LOGGER) as parser:
        with open(onnx_path, "rb") as model:
            parser.parse(model.read())

    # Create an optimization profile
    profile = builder.create_optimization_profile()

    # set the dimensions of the input tensor
    profile.set_shape("vi_img", (1, 1, 256, 320), (1, 1, 256, 320), (1, 1, 256, 320))
    profile.set_shape("ir_img", (1, 1, 256, 320), (1, 1, 256, 320), (1, 1, 256, 320))

    # Add the optimization profile
    config = builder.create_builder_config()
    config.add_optimization_profile(profile)

    # Set the maximum workspace size
    config.max_workspace_size = 1 << 30  # For example, 1GB

    # Build the engine
    engine = builder.build_engine(network, config)

    with open(engine_path, "wb") as f:
        f.write(engine.serialize())


onnx2trt("./SemLA_onnx_tensorrt.onnx", "./SemLA_tensorrt.engine")
