from pydantic import BaseModel


class Edit(BaseModel):
    file_name: str
    line_number: int  # indexed by 0
    line_content: str
    new_line_content: str


class Patch(BaseModel):
    edits: list[Edit]


class ChangedFile(BaseModel):
    file_name: str
    old_content: str
    new_content: str


class ChangedFiles(BaseModel):
    files: list[ChangedFile]


def apply_edits(old_content: str, edits: list[Edit]):
    """
    Apply the patch to old_content. For each Edit in the patch, the line at the given
    index is replaced with the new_line_content. If the edit refers to a line that does
    not yet exist, the list is extended with empty lines until the index is reached.
    """
    # Make a copy so we don't mutate the original list
    new_content = old_content.split("\n")
    for edit in edits:
        if edit.line_number < len(new_content):
            new_content[edit.line_number] = edit.new_line_content
        else:
            # Extend the list with empty strings until we can add the new line.
            new_content.extend([""] * (edit.line_number - len(new_content)))
            new_content.append(edit.new_line_content)
    return "\n".join(new_content)
