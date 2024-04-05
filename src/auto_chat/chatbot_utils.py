from src.auto_chat.pipline import Pipeline
from typing import List, Optional


TITLE_MARKDOWN = (
    """
    <br><br>
    # &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; AutoChat
    """
)

EXAMPLES = [
    "What does Fusion 360 do?",
    "What's the difference between AutoCAD and Revit?",
    "Does AutoCAD LT do 3d?",
    "What's the latest release for Maya?",
    "Can I use fusion 360 on a Mac?"
]


def get_metadata_html(query: Optional[str], chunks: List):
    metadata_html = ""
    if query:
        metadata_html += f"========================================IR Query========================================"
        metadata_html += f"\n{str(query)}\n\n"
        metadata_html = f"<pre> {metadata_html} </pre>"

    for i, chunk in enumerate(chunks):
        metadata_html += f"Chunk {i + 1}:"
        metadata_html += f"\n{str(chunk).replace('<', '&lt;').replace('>', '&gt;')}\n"

    metadata_html = metadata_html.replace('\n', "<br>")
    # metadata_html = f"<pre> {metadata_html} </pre>"
    return metadata_html

