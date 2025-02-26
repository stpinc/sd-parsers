"""
Find stealth (LSB steganography) metadata embedded in a successfully parsed folder of images, and copies it to the EXIF metadata.
Does not remove the embedded stealth data.
Note: This will change the modified time on the file.
usage: python3 convert_stealth_metadata.py <a file or directory>
"""

from pathlib import Path
from PIL import Image
from PIL import UnidentifiedImageError
from PIL.PngImagePlugin import PngInfo
from sd_parsers import ParserManager
from tqdm import tqdm
import json
import logging
import sys

parser_manager = ParserManager()

# TODO: 
# It may be worth figuring out how to delete the stealth data from the image, to avoid processing and re-saving every time.
# Another option is to put something in the metadata indicating it's been processed already.
# Right now, this will re-write the metadata of files that may have already been processed.
def parse(filename):
    prompt_info = parser_manager.parse(filename)
    try:
        if prompt_info.metadata["Decoded from Stealth Metadata"] == "True":
            metadata = PngInfo()
            match prompt_info.raw_parameters["Software"]:
                case "NovelAI":
                    for k, v in prompt_info.raw_parameters.items():
                        metadata.add_text(str(k), str(v))

                    # Software is picky about the JSON being properly formatted on the Comment field, do this one 
                    # properly instead of just copying the string over
                    metadata.add_text(
                        "Comment", json.dumps((prompt_info.raw_parameters["Comment"]))
                    )
                case "AUTOMATIC1111":
                    metadata.add_text(
                        "parameters", prompt_info.raw_parameters["parameters"]
                    )
            with Image.open(filename) as new_image:
                new_image.save(filename, pnginfo=metadata)
                return True

    except UnidentifiedImageError:
        pass

    except (KeyError, AttributeError):
        # print(f"{filename} did not contain any hidden metadata.")
        pass

    except Exception as e:
        logging.exception(f"Error reading {filename}: {e}")


def main(file):
    # If we're dealing with a directory, grab a list of all files within (including subfolders) and iterate over them.
    if Path(file[0]).is_dir():
        file_glob = Path(file[0]).rglob("*")
        num_items = len(list(file_glob))

        with tqdm(total=num_items) as progress_bar:
            progress_bar.set_description("Searching")
            conv_items: int = 0

            for filename in Path(file[0]).rglob("*"):
                progress_bar.update(1)

                if filename.is_file():
                    try:
                        if parse(filename):
                            progress_bar.write(f"Converted {str(filename)}")
                            conv_items += 1

                    except Exception:
                        logging.exception(f"some kind of error reading {filename}")

            progress_bar.set_description("Complete")

        progress_bar.write(f"Done. Converted {conv_items} items.")

    # If it's not a directory, just do a simple single-file parse
    else:
        try:
            if parse(file[0]):
                logging.info(f"Converted {str(file[0])}")

        except Exception:
            logging.exception(f"some kind of error reading {file[0]}")


if __name__ == "__main__":
    if sys.argv[1:]:
        main(sys.argv[1:])
    else:
        print("usage: convert_stealth_metadata.py <a file or directory>")
