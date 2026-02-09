import onnx

m = onnx.load("models/detection/icms1/w_sami.onnx")

print("Graph outputs:")
for o in m.graph.output:
    print("  declared:", o.name)

print("\nAll node outputs:")
for n in m.graph.node:
    for out in n.output:
        print("  node:", n.op_type, "->", out)