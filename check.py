import onnx

m = onnx.load("/home/edgeai/code/edgeai-benchmark/pha2_models/keypoints/v1/0919_384_640.onnx")

print("Graph outputs:")
for o in m.graph.output:
    print("  declared:", o.name)

print("\nAll node outputs:")
for n in m.graph.node:
    for out in n.output:
        print("  node:", n.op_type, "->", out)