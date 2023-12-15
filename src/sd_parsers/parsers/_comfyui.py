"""Parser for images generated by ComfyUI or similar."""
import json
import logging
from collections import defaultdict
from contextlib import suppress
from typing import Any, Dict, Generator, List, Optional, Set, Tuple

from PIL.Image import Image

from .._models import Model, Prompt, Sampler
from .._parser import Generators, Parser, ParseResult
from .._prompt_info import PromptInfo
from ..exceptions import ParserError

logger = logging.getLogger(__name__)

SAMPLER_PARAMS = {"sampler_name", "steps", "cfg"}
POSITIVE_PROMPT_KEYS = ["text", "positive"]
NEGATIVE_PROMPT_KEYS = ["text", "negative"]
IGNORE_LINK_TYPES_PROMPT = ["CLIP"]


class ComfyUIParser(Parser):
    """Parser for images generated by ComfyUI"""

    @property
    def generator(self):
        return Generators.COMFYUI

    def read_parameters(self, image: Image, use_text: bool = True):
        if image.format != "PNG":
            return None, None

        try:
            metadata = image.text if use_text else image.info  # type: ignore

            prompt = json.loads(metadata["prompt"])
            workflow = json.loads(metadata["workflow"])
        except (KeyError, json.JSONDecodeError, TypeError) as error:
            return None, error

        return PromptInfo(self, {"prompt": prompt, "workflow": workflow}), None

    def parse(self, parameters: Dict[str, Any], _) -> ParseResult:
        try:
            prompt = parameters["prompt"]
            workflow = parameters["workflow"]
        except KeyError as error:
            raise ParserError("error reading parameters") from error

        samplers, metadata = ImageContext.extract(self, prompt, workflow)
        return samplers, metadata


class ImageContext:
    parser: ComfyUIParser
    prompt: Dict[int, Any]
    links: Dict[int, Dict[int, Set[str]]]
    processed_nodes: Set[int]

    def __init__(self, parser: ComfyUIParser, prompt, workflow):
        self.parser = parser
        self.processed_nodes = set()

        # ensure that prompt keys are integers
        try:
            self.prompt = {int(k): v for k, v in prompt.items()}
        except (AttributeError, ValueError) as error:
            raise ParserError("prompt has unexpected format") from error

        # build links dictionary (dict[input_id, dict[output_id, set[link_type]]])
        self.links = defaultdict(lambda: defaultdict(set))
        try:
            for _, output_id, _, input_id, _, link_type in workflow["links"]:
                self.links[int(input_id)][int(output_id)].add(link_type)
        except (TypeError, KeyError, ValueError) as error:
            raise ParserError("workflow has unexpected format") from error

    @classmethod
    def extract(
        cls, parser: ComfyUIParser, prompt: Any, links: Any
    ) -> Tuple[List[Sampler], Dict[Tuple[str, int], Dict[str, Any]]]:
        """Extract samplers with their child parameters aswell as metadata"""
        context = cls(parser, prompt, links)
        samplers = []
        metadata = {}

        # Pass 1: get samplers and related data
        for node_id, node in context.prompt.items():
            sampler = context._try_get_sampler(node_id, node)
            if sampler:
                samplers.append(sampler)

        # Pass 2: put information from unprocessed nodes into metadata
        for node_id, node in context.prompt.items():
            if node_id in context.processed_nodes:
                continue

            with suppress(KeyError, AttributeError):
                inputs = {
                    key: value
                    for key, value in node["inputs"].items()
                    if not isinstance(value, list)
                }
                if inputs:
                    metadata[node["class_type"], node_id] = inputs

        return samplers, metadata

    def _traverse(
        self, node_id: int, ignored_link_types: Optional[List[str]] = None
    ) -> Generator[Tuple[int, Any], Optional[bool], None]:
        """Traverse backwards through node tree, starting at a given node_id"""
        visited = set()
        ignore_links = set(ignored_link_types) if ignored_link_types else set()

        def traverse_inner(
            node_id: int, depth: int = 0
        ) -> Generator[Tuple[int, Any], Optional[bool], None]:
            visited.add(node_id)

            with suppress(KeyError):
                recurse = yield node_id, self.prompt[node_id]
                if recurse is False:
                    return

            with suppress(KeyError, RecursionError):
                for link_id, link_types in self.links[node_id].items():
                    if link_id not in visited and link_types - ignore_links:
                        logger.debug(
                            "traverse %d->%d, %s%s", node_id, link_id, "\t" * depth, link_types
                        )
                        yield from traverse_inner(link_id, depth + 1)

        yield from traverse_inner(node_id)

    def _get_prompts(self, initial_node_id: int, text_keys: List[str]) -> List[Prompt]:
        """Get all prompts reachable from a given node_id."""
        logger.debug("looking for prompts: %d", initial_node_id)

        prompts = []

        def check_inputs(node_id: int, inputs: Dict) -> bool:
            found_prompt = False
            for key in text_keys:
                try:
                    text = inputs[key]
                except KeyError:
                    continue

                if isinstance(text, str):
                    logger.debug("found prompt %s#%d: %s", key, node_id, text)
                    prompts.append(Prompt(value=text.strip(), prompt_id=node_id))
                    found_prompt = True

            if found_prompt:
                self.processed_nodes.add(node_id)
                return False
            return True

        prompt_iterator = self._traverse(initial_node_id, IGNORE_LINK_TYPES_PROMPT)
        with suppress(StopIteration):
            node_id, node = next(prompt_iterator)
            while True:
                recurse = True
                try:
                    inputs = node["inputs"]
                except KeyError:
                    pass
                else:
                    recurse = check_inputs(node_id, inputs)

                node_id, node = prompt_iterator.send(recurse)

        return prompts

    def _get_model(self, initial_node_id: int) -> Optional[Model]:
        """Get the first model reached from the given node_id"""
        logger.debug("looking for model: #%s", initial_node_id)

        for node_id, node in self._traverse(initial_node_id):
            try:
                inputs = node["inputs"]
                ckpt_name = inputs["ckpt_name"]
            except KeyError:
                pass
            else:
                self.processed_nodes.add(node_id)
                logger.debug("found model #%d: %s", node_id, ckpt_name)

                model = Model(model_id=node_id, name=ckpt_name)

                with suppress(KeyError):
                    model.parameters["config_name"] = inputs["config_name"]

                return model

        return None

    def _try_get_sampler(self, node_id: int, node):
        """Test if this node could contain sampler data"""
        try:
            if not SAMPLER_PARAMS.issubset(node["inputs"]):
                return None
        except (KeyError, TypeError):
            return None

        logger.debug("found sampler #%d", node_id)
        self.processed_nodes.add(node_id)

        inputs = dict(node["inputs"])

        # Sampler parameters
        sampler_name = inputs.pop("sampler_name")
        sampler_parameters = self.parser.normalize_parameters(
            (key, value) for key, value in inputs.items() if not isinstance(value, list)
        )

        # Sampler
        sampler = Sampler(
            sampler_id=node_id,
            name=sampler_name,
            parameters=sampler_parameters,
        )

        # Model
        with suppress(KeyError, ValueError):
            model_id = int(inputs["model"][0])
            sampler.model = self._get_model(model_id)

        # Prompt
        with suppress(KeyError, ValueError):
            positive_prompt_id = int(inputs["positive"][0])
            sampler.prompts = self._get_prompts(positive_prompt_id, POSITIVE_PROMPT_KEYS)

        # Negative Prompt
        with suppress(KeyError, ValueError):
            negative_prompt_id = int(inputs["negative"][0])
            sampler.negative_prompts = self._get_prompts(negative_prompt_id, NEGATIVE_PROMPT_KEYS)

        return sampler
