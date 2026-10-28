# SPDX-License-Identifier: MIT  -- QW
# 2026-08-04  ZO graph translation: make an Onnx4Deeploy ZO graph consumable by TrainDeeploy's
# training frontend. The Onnx4Deeploy ZO graphs target base-Deeploy's inference frontend (mezo
# custom-domain PerturbRademacher, standard 5-in/1-out eval BatchNormalization, single misnamed
# SCE output), which TrainDeeploy's training deployer rejects. This pass rewrites them to the
# TrainDeeploy-native op forms. Applied in run_zo_codegen before deployer.prepare().
#
# Transforms:
#  1. Strip the `mezo` (and ai.onnx.contrib) domain from Perturb* nodes -> default domain, so the
#     PULPMapping['PerturbRademacher'] lookup matches (Deeploy keys on op_type) and ORT's failing
#     shape-inference on the unregistered custom op is bypassed by pre-annotated shapes.
#  2. BatchNormalization (5-in, 1-out eval) -> BatchNormInternal (5-in, 5-out; TrainDeeploy's
#     frozen-stats forward BN via BN_FROZEN_STATS). Adds the 4 stat outputs it expects
#     (updated_running_mean/var, saved_mean, saved_inv_std) as dangling [C] tensors.
#  3. SoftmaxCrossEntropyLoss single-output (loss, misnamed `log_prob`, mis-shaped [1,K]) ->
#     canonical 2-output form (outputs[0]=scalar loss [1], outputs[1]=log_prob [B,K]); both are
#     graph outputs (mirrors the BP graph). The runner reads outputs[0] = the scalar loss.
#  4. Drop the mezo/contrib opset imports; run onnx shape inference to fill the rest.
import onnx
from onnx import shape_inference, helper, TensorProto


def prep_zo_graph(in_path: str, out_path: str) -> dict:  # -- QW
    m = onnx.load(in_path)
    g = m.graph
    report = {"perturb_domain_stripped": 0, "bn_converted": 0, "sce_split": 0}

    for n in g.node:
        if n.op_type.startswith("Perturb") and n.domain:
            n.domain = ""
            report["perturb_domain_stripped"] += 1

    init = {i.name: list(i.dims) for i in g.initializer}
    # Full shape lookup: the BN scale may be a *perturbed* tensor (output of a
    # PerturbRademacher node), so it is neither an initializer nor a graph input.
    # Fall back through value_info / inputs, and finally trace a Perturb producer
    # back to its clean initializer, so C resolves to the real channel count
    # instead of 0 (which collapses the stat outputs to shape [0] -> infeasible
    # tiler geometric constraint). -- QW
    shape_of = dict(init)
    for vi in list(g.value_info) + list(g.input) + list(g.output):
        shape_of.setdefault(vi.name, [d.dim_value for d in vi.type.tensor_type.shape.dim])
    producer = {o: n for n in g.node for o in n.output}

    def _channels(scale_name):  # -- QW
        s = shape_of.get(scale_name)
        if s and s[0] > 0:
            return s[0]
        p = producer.get(scale_name)
        if p is not None and p.op_type.startswith("Perturb") and p.input:
            s = shape_of.get(p.input[0])
            if s and s[0] > 0:
                return s[0]
        return 0

    for n in g.node:
        if n.op_type == "BatchNormalization":
            C = _channels(n.input[1])  # scale/gamma shape = [C]
            base = n.output[0]
            for e in (f"{base}_upmean", f"{base}_upvar", f"{base}_savedmean", f"{base}_savedinvstd"):
                n.output.append(e)
                g.value_info.append(helper.make_tensor_value_info(e, TensorProto.FLOAT, [C]))
            n.op_type = "BatchNormInternal"
            n.domain = ""
            if not any(a.name == "training_mode" for a in n.attribute):
                n.attribute.append(helper.make_attribute("training_mode", 1))
            report["bn_converted"] += 1

    vmap = {vi.name: [d.dim_value for d in vi.type.tensor_type.shape.dim]
            for vi in list(g.value_info) + list(g.output) + list(g.input)}
    for n in g.node:
        if n.op_type == "SoftmaxCrossEntropyLoss":
            lshape = vmap.get(n.input[0], [1, 9])
            B, K = lshape[0], lshape[1]
            del n.output[:]
            n.output.extend(["zo_loss", "zo_log_prob"])
            g.value_info.append(helper.make_tensor_value_info("zo_log_prob", TensorProto.FLOAT, [B, K]))
            del g.output[:]
            # loss is a 0-d (rank-0) scalar, matching BP's working SoftmaxCrossEntropyLoss
            # output (shape []). Declaring it [1] gives it a dim_0 that the SCE geometric
            # tile constraint never registers -> tiler KeyError output_0_dim_0. -- QW
            g.output.append(helper.make_tensor_value_info("zo_loss", TensorProto.FLOAT, []))
            g.output.append(helper.make_tensor_value_info("zo_log_prob", TensorProto.FLOAT, [B, K]))
            report["sce_split"] += 1

    # --- Liveness fix for the in-place ZO update graph. -- QW
    #
    # The Onnx4Deeploy zo_update graph is 22 *in-place* PerturbRademacher nodes
    # (input name == output name == a weight initializer) with NO graph outputs.
    # With 0 graph outputs TrainDeeploy's scheduler prunes every node as dead
    # code -> the tiler builds an empty schedule -> _setupObjective indexes
    # self._objectives[0] on an empty list -> IndexError. Fix: give each in-place
    # perturb a DISTINCT '<weight>_updated' output tensor and register it as a
    # graph output so the node stays live and the tiler gets a per-node
    # objective. The weights stay as initializers (baked, loadable), so the
    # update graph codegens standalone. The '_updated' suffix also lets
    # build_shared_buffer_maps alias the update output onto the training weight
    # buffer *iff* those weights are later exposed as training graph inputs
    # (they are initializers in the current fixtures, so no aliasing happens yet
    # -- see the buffer-sharing note in the report / ZO_FINDING.md §2c).
    #
    # Only fires on in-place Perturb* nodes, so the BP path (no Perturb nodes)
    # and the zo_train graph (perturb outputs are consumed, not in-place graph
    # outputs) are both untouched. -- QW
    perturbed = [n for n in g.node if n.op_type.startswith("Perturb")]
    if perturbed:
        init_by_name = {i.name: i for i in g.initializer}
        input_names = {i.name for i in g.input}
        report["update_outputs_added"] = 0
        for n in perturbed:
            wname = n.input[0]
            # in-place perturb (output name == input name) with no other consumer
            # -> rebind to a distinct '<weight>_updated' graph output.
            if len(n.output) == 1 and n.output[0] == wname:
                consumed = any(wname in other.input for other in g.node if other is not n)
                if not consumed and wname not in input_names:
                    new_out = f"{wname}_updated"
                    n.output[0] = new_out
                    dims = list(init_by_name[wname].dims) if wname in init_by_name else []
                    g.output.append(helper.make_tensor_value_info(new_out, TensorProto.FLOAT, dims))
                    report["update_outputs_added"] += 1

    keep = [op for op in m.opset_import if op.domain not in ("mezo", "ai.onnx.contrib")]
    del m.opset_import[:]
    m.opset_import.extend(keep)
    try:
        m = shape_inference.infer_shapes(m, strict_mode=False, data_prop=True)
        report["shape_infer"] = "ok"
    except Exception as e:  # noqa
        report["shape_infer"] = f"err: {str(e)[:120]}"
    onnx.save(m, out_path)
    return report


if __name__ == "__main__":
    import sys, json
    print(json.dumps(prep_zo_graph(sys.argv[1], sys.argv[2]), indent=2))
