import torch
from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
    get_symmetric_quantization_config,
    XNNPACKQuantizer,
)
from torch.export.exported_program import ExportedProgram
from itertools import islice
from torch.ao.quantization.quantize_pt2e import convert_pt2e
from torch.ao.quantization.quantize_pt2e import prepare_pt2e
import nncf.experimental.torch.fx as nncf_fx  # type: ignore[import-untyped]
import nncf

def get_xnnpack_quantizer(kwargs):
    quantizer = XNNPACKQuantizer()
    # if we set is_per_channel to True, we also need to add out_variant of quantize_per_channel/dequantize_per_channel
    kwargs["is_dynamic"] = False
    if "is_per_channel" not in kwargs:
        kwargs["is_per_channel"] = False
    operator_config = get_symmetric_quantization_config(
        **kwargs
    )
    quantizer.set_global(operator_config)
    return quantizer


def quantize_model(model, example_args, calibration_dataset, transform_fn):
    aten_dialect: ExportedProgram = torch.export.export_for_training(model, example_args, strict=True)
    quantizer = get_xnnpack_quantizer({})
    m = prepare_pt2e(aten_dialect.module(), quantizer)
    # calibration

    subset_size = 300
    batch_size = calibration_dataset.batch_size or 1
    subset_size = (subset_size // batch_size) + int(subset_size % batch_size > 0)

    for sample in islice(calibration_dataset, subset_size):
        m(transform_fn(sample))
    quantized_model = convert_pt2e(m)
    return quantized_model


def quantize_model_nncf(model, example_args, calibration_dataset, transform_fn):
    aten_dialect: ExportedProgram = torch.export.export_for_training(model, example_args, strict=True)
    quantizer = get_xnnpack_quantizer({})
    #quantize_pt2e_kwargs = quantize_pt2e_kwargs or {}
    quantize_pt2e_kwargs = {}
    quantize_pt2e_kwargs["fold_quantize"] = True

    subset_size = 300
    batch_size = calibration_dataset.batch_size or 1
    subset_size = (subset_size // batch_size) + int(subset_size % batch_size > 0)

    return nncf_fx.quantize_pt2e(
        aten_dialect.module(),
        quantizer,
        subset_size=subset_size,
        calibration_dataset=nncf.Dataset(calibration_dataset, transform_fn),
        **quantize_pt2e_kwargs
    )