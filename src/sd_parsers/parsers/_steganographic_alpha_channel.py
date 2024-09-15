"""Parser for images containing metadata inside the Alpha channel."""
import logging
import gzip
import json

from typing import Any, Dict

from PIL.Image import Image

from sd_parsers.data import Generators, Model, Prompt, Sampler
from sd_parsers.exceptions import MetadataError, ParserError
from sd_parsers.parser import Parser, ParseResult
from sd_parsers.parsers._novelai import NovelAIParser
from sd_parsers.parsers._automatic1111 import AUTOMATIC1111Parser

logger = logging.getLogger(__name__)
STEALTH_HEADER = "stealth_pngcomp"

class SteganographicAlphaChannelParser(Parser):
    """
    Parser for PNG files to determine whether they have metadata stored in the Alpha channel. \n
    This should run last whenever possible, as it's computationally expensive compared to reading EXIF data.
    """
    
    class LSBExtractor():
        def __init__(self, img):
            self.data = list(img.getdata())
            self.width, self.height = img.size
            self.data = [
                self.data[i * self.width : (i + 1) * self.width]
                for i in range(self.height)
            ]
            self.dim = 4
            self.bits = 0
            self.byte = 0
            self.row = 0
            self.col = 0

        def _extract_next_bit(self):
            if self.row < self.height and self.col < self.width:
                bit = self.data[self.row][self.col][self.dim - 1] & 1
                self.bits += 1
                self.byte <<= 1
                self.byte |= bit
                self.row += 1
                if self.row == self.height:
                    self.row = 0
                    self.col += 1

        def get_one_byte(self):
            while self.bits < 8:
                self._extract_next_bit()
            byte = bytearray([self.byte])
            self.bits = 0
            self.byte = 0
            return byte

        def get_next_n_bytes(self, n):
            bytes_list = bytearray()
            for _ in range(n):
                byte = self.get_one_byte()
                if not byte:
                    break
                bytes_list.extend(byte)
            return bytes_list

        def read_32bit_integer(self):
            bytes_list = self.get_next_n_bytes(4)
            if len(bytes_list) == 4:
                integer_value = int.from_bytes(bytes_list, byteorder="big")
                return integer_value
            else:
                return None

    @property
    def generator(self):
        return Generators.STEGANOGRAPHIC_ALPHA

    def read_parameters(self, image: Image, use_text: bool = True):
        """
        Determine whether the image has the stealth header. If it does, then check whether it has JSON data or just plain text. \n
        Afterwards, store the correct parser to use via parsing_context.
        """
        parameters = {}
        parsing_context = {}

        try:
            extractor = self.LSBExtractor(image)
            if image.format == "PNG" and image.mode == "RGBA":
                valid_json = False
                try:
                    read_magic = extractor.get_next_n_bytes(
                        len(STEALTH_HEADER)
                    ).decode("utf-8")
                    assert (
                        STEALTH_HEADER == read_magic
                    ), f"Header \"{read_magic}\" does not match the expected value of \"{STEALTH_HEADER}\"!"
                except Exception as error:
                    raise MetadataError("no matching metadata") from error
                else:
                    read_len = (extractor.read_32bit_integer() // 8)
                    raw_bytes = extractor.get_next_n_bytes(read_len)
                    decompressed_bytes = gzip.decompress(raw_bytes).decode("utf-8") 
                    logger.debug(f"\"{read_magic}\" was found in the header, doing a full decode of all bytes.")
                    try:
                        logger.debug("Checking for NovelAI JSON.")
                        json_data = json.loads(decompressed_bytes)
                        if json_data:
                            description = json_data["Description"]
                            software = json_data["Software"]
                            source = json_data["Source"]
                            comment = json.loads(json_data["Comment"])
                            valid_json = True
                            parsing_context["Source"] = "NovelAI Stealth Metadata"
                            parameters = {
                                "Comment": comment,
                                "Description": description,
                                "Software": software,
                                "Source": source,
                            }
                            return parameters, parsing_context

                    except json.decoder.JSONDecodeError:
                        logger.debug("No JSON found, probably not NovelAI.")

                if not valid_json:
                    parsing_context["Source"] = "AUTOMATIC1111 StealthPNG"
                    return {"parameters": decompressed_bytes}, parsing_context

            else:
                raise MetadataError("unsupported image format", image.format)

        except (KeyError, ValueError) as error:
            raise MetadataError("no matching metadata") from error

    def parse(self, parameters: Dict[str, Any], parsing_context: Any) -> ParseResult:
        """
        Send the metadata from the previous step to the correct parser. \n
        The data is sent directly to the other parser's parse() method.
        """
        logger.debug(f"Attempting to parse [{parsing_context["Source"]}] metadata.")

        match parsing_context["Source"]:
            case "NovelAI Stealth Metadata":
                try:
                    parameters["Comment"].update({"decoded_from_stealth_metadata": True})
                    return NovelAIParser.parse(self, parameters, Any)
                except KeyError as error:
                    raise ParserError("error reading parameter values") from error

            case "AUTOMATIC1111 StealthPNG":
                try:
                    parameters["parameters"] += ", Decoded from Stealth Metadata: True"
                    print("steg parser reading")
                    return AUTOMATIC1111Parser.parse(self, parameters, Any)
                except (KeyError, ValueError) as error:
                    raise ParserError("error reading parameter string") from error
