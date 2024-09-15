import pytest
from PIL import Image
from sd_parsers.data import Model, Prompt, Sampler
from sd_parsers.parsers import SteganographicAlphaChannelParser, _steganographic_alpha_channel
from sd_parsers.parsers import AUTOMATIC1111Parser, _automatic1111

from tests.tools import RESOURCE_PATH

PARAMETERS = (
    "a circle\nNegative prompt: a square\n"
    "Steps: 20, Sampler: Euler, CFG scale: 7, Seed: 2015833630, Size: 64x64, "
    "Model hash: 15012c538f, Model: realisticVisionV51_v51VAE"
)

MODEL = Model(name="realisticVisionV51_v51VAE", hash="15012c538f")
PROMPTS = [Prompt(1, "a circle")]
NEGATIVE_PROMPTS = [Prompt(1,"a square")]

SAMPLER = Sampler(
    name="Euler",
    parameters={'scheduler': 'Automatic', 'cfg_scale': '7', 'seed': '2015833630', 'steps': '20'},
    model=MODEL,
    prompts=PROMPTS,
    negative_prompts=NEGATIVE_PROMPTS,
)

# Cropping the image to 1x1 for the tests doesn't work since it needs enough bytes to decode the data.

testdata = [
    pytest.param(
        "a1111_stealth.png",
        (
            [SAMPLER],
            {
                'Clip skip': '2',
                'Decoded from Stealth Metadata': 'True',
                'Size': '64x64',
                'Version': 'v1.9.4-169-ga30b19dd',
            },
        ),
        id="a1111 stealth",
    )
]


@pytest.mark.parametrize("filename, expected", testdata)
def test_parse(filename: str, expected):
    (
        expected_samplers,
        expected_metadata,
    ) = expected

    parser = SteganographicAlphaChannelParser()
    with Image.open(RESOURCE_PATH / "parsers/SteganographicAlphaChannel" / filename) as image:
        image_data = parser.read_parameters(image)


    samplers, metadata = parser.parse(*image_data)

    assert image_data is not None
    assert samplers == expected_samplers
    assert metadata == expected_metadata


def test_civitai_hashes():
    parameters = PARAMETERS + ', Hashes: {"vae": "c6a580b13a", "model": "15012c538f"}'

    # since the metadata is in the A111 format, hand it off to that parser
    info_index, sampler_info, metadata = _automatic1111.get_sampler_info(parameters.split("\n"))

    print(sampler_info)
    print(metadata)
    assert info_index == 2
    assert sampler_info == {
        "CFG scale": "7",
        "Sampler": "Euler",
        "Seed": "2015833630",
        "Steps": "20",
    }
    assert metadata == {
        "Model": "realisticVisionV51_v51VAE",
        "Model hash": "15012c538f",
        "Size": "64x64",
        "civitai_hashes": {"model": "15012c538f", "vae": "c6a580b13a"},
    }
