from enum import Enum


class Format(Enum):
    MARKDOWN = "markdown"


SUPPORTED_FORMATS = {
    Format.MARKDOWN.value: [
        # Split along Markdown headings (starting with level 2)
        "\n#{1,6} ",
        # Split along Markdown code blocks
        "```\n",
        # Horizontal lines
        "\n\\*\\*\\*+\n",
        "\n---+\n",
        "\n___+\n",
        # Split along Markdown lists
        "\n\n",
        "\n",
        " ",
        "",
    ]
}


def get_separators(format: Format):
    """
    Get the separators for a given format.
    """
    separators = SUPPORTED_FORMATS.get(format)
    if separators is None:
        raise KeyError(f"No supported separators for format of type: {format}.")
    return separators
