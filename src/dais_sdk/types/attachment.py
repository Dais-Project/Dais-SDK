from typing import Literal
from pydantic import BaseModel

class Base64Source(BaseModel):
    type: Literal["base64"] = "base64"
    mime_type: str
    data: str

class UrlSource(BaseModel):
    type: Literal["url"] = "url"
    url: str

type FileSource = Base64Source | UrlSource

class ImageAttachment(BaseModel):
    type: Literal["image"] = "image"
    source: FileSource

class DocumentAttachment(BaseModel):
    type: Literal["document"] = "document"
    source: FileSource

class AudioAttachment(BaseModel):
    type: Literal["audio"] = "audio"
    source: FileSource

class VideoAttachment(BaseModel):
    type: Literal["video"] = "video"
    source: FileSource

type Attachment = ImageAttachment | DocumentAttachment | AudioAttachment | VideoAttachment

__all__ = [
    "FileSource",
    "Attachment",

    "Base64Source",
    "UrlSource",
    "ImageAttachment",
    "DocumentAttachment",
    "AudioAttachment",
    "VideoAttachment",
]
