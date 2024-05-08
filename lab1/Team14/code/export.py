import torch
from torch._export import capture_pre_autograd_graph
from torch.export import export, ExportedProgram
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from torch import nn
import pickle
import io
import pickle
from torch.export import dynamic_dim
from executorch.exir import ExecutorchBackendConfig, ExecutorchProgramManager, EdgeProgramManager, to_edge
from executorch.exir.passes import MemoryPlanningPass
import executorch.exir as exir
NUM_CLASSES = 10


if __name__ == '__main__':
    # contents = CPU_Unpickler(f).load()
    model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
    weight_path = '/home/aa35037123/Wesley/edge_ai/lab1/mobilenet.pt'
    # with torch.loading_context(map_location='cpu'):
    # weights = CPU_Unpickler(weight_path).load()
    # Load the state_dict to the model
    model.load_state_dict(torch.load(weight_path, map_location=torch.device('cpu')))
    model.to('cpu')
    model.eval()
    # model.load_state_dict(torch.load('/home/aa35037123/Wesley/edge_ai/lab1/mobilenet.pt'), map_location=torch.device('cpu'))
    # image size is 224, batch is 1
    example_args = (torch.randn(1, 3, 224, 224),)
    pre_autograd_aten_dialect = capture_pre_autograd_graph(model, example_args)
    print("Pre-Autograd ATen Dialect Graph")
    print(pre_autograd_aten_dialect)
    aten_dialect: ExportedProgram = torch.export.export(pre_autograd_aten_dialect, example_args)
    print("ATen Dialect Graph")
    print(aten_dialect)

    
    edge_program: EdgeProgramManager = exir.to_edge(aten_dialect)
    print("Edge Dialect Graph")
    print(edge_program.exported_program())



    # # Optionally do delegation:
    # # edge_program = edge_program.to_backend(CustomBackendPartitioner)
    executorch_program: ExecutorchProgramManager = edge_program.to_executorch(
        ExecutorchBackendConfig(
            passes=[],  # User-defined passes
        )
    )

    with open("mobilenet.pte", "wb") as file:
        file.write(executorch_program.buffer)

