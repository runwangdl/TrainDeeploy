from Deeploy.Targets.GAP9.Platform import GAP9Platform, GAP9Mapping
print("GAP9Platform import OK")
for k in ["BatchNormInternal","PerturbRademacher","RQSPerturbRademacher","Quant","Dequant","RequantShift","SoftmaxCrossEntropyLoss","SGD"]:
    present = type(GAP9Mapping.get(k)).__name__ if k in GAP9Mapping else "MISSING"
    print(f"  mapping[{k}] = {present}")
