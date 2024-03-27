import torch
import torchvision.models as models

from torch.export import export
from torchvision.models.mobilenetv2 import MobileNet_V2_Weights
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.exir import to_edge
from torch import nn

NUM_CLASSES = 10

if __name__ == "__main__":
    mobilenet_v2 = models.mobilenetv2.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT).eval()
    mobilenet_v2.classifier[1] = nn.Linear(mobilenet_v2.classifier[1].in_features, NUM_CLASSES)
    weight_path = '/home/aa35037123/Wesley/edge_ai/lab1/mobilenet.pt'
    
    mobilenet_v2.load_state_dict(torch.load(weight_path, map_location=torch.device('cpu')))
    mobilenet_v2.to('cpu')
    
    sample_inputs = (torch.randn(1, 3, 224, 224), )

    edge = to_edge(export(mobilenet_v2, sample_inputs))

    edge = edge.to_backend(XnnpackPartitioner)
    print(edge.exported_program().graph_module)
    exec_prog = edge.to_executorch()

    with open("xnn_mobilenet.pte", "wb") as file:
        file.write(exec_prog.buffer)