"""Parser for images generated by NovelAI or similar."""
import copy
import json
import re
from contextlib import suppress
from typing import Any, Dict

from PIL.Image import Image

from .._models import Model, Prompt, Sampler
from .._parser import Generators, Parser, ParseResult, pop_keys
from .._prompt_info import PromptInfo
from ..exceptions import ParserError

SAMPLER_PARAMS = ["seed", "strength", "noise", "scale"]


class NovelAIParser(Parser):
    """parser for images generated by NovelAI"""

    @property
    def generator(self):
        return Generators.NOVELAI

    def read_parameters(self, image: Image, use_text: bool = True):
        if image.format != "PNG":
            return None

        metadata = image.text if use_text else image.info  # type: ignore
        try:
            description = metadata["Description"]
            software = metadata["Software"]
            source = metadata["Source"]
            comment = json.loads(metadata["Comment"])
        except KeyError:
            return None
        except (json.JSONDecodeError, TypeError) as error:
            raise ParserError("error reading metadata") from error

        if software != "NovelAI":
            return None

        return PromptInfo(
            self,
            {
                "Comment": comment,
                "Description": description,
                "Software": software,
                "Source": source,
            },
        )

    def parse(self, parameters: Dict[str, Any], _) -> ParseResult:
        try:
            metadata = copy.deepcopy(parameters["Comment"])
            params = parameters["Description"]
            source = parameters["Source"]
        except KeyError as error:
            raise ParserError("error reading parameter values") from error

        try:
            sampler = {
                "name": metadata.pop("sampler"),
                "parameters": self.normalize_parameters(pop_keys(SAMPLER_PARAMS, metadata)),
            }
        except KeyError as error:
            raise ParserError("no sampler found") from error

        sampler["prompts"] = [Prompt(params.strip())]

        with suppress(KeyError):
            sampler["negative_prompts"] = [Prompt(metadata.pop("uc"))]

        # model
        match = re.fullmatch(r"^(.*?)\s+([A-Z0-9]+)$", source)
        if match:
            model_name, model_hash = match.groups()
            sampler["model"] = Model(name=model_name, model_hash=model_hash)

        metadata = self.normalize_parameters(metadata)

        return [Sampler(**sampler)], metadata
