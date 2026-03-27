"""Test suite for skill archive parsing in dais_sdk.skill."""

from __future__ import annotations

import io
import zipfile
from pathlib import PurePosixPath

import pytest

from dais_sdk.skill import Skill, SkillMd, SkillParser, SkillResource
from dais_sdk.types.exceptions import InvalidSkillArchiveError


def _make_zip_bytes(entries: dict[str, str | bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        for name, content in entries.items():
            data = content.encode("utf-8") if isinstance(content, str) else content
            zf.writestr(name, data)
    return buffer.getvalue()


class TestSkillParser:
    def test_find_skill_root_from_first_level_directory(self):
        paths = [
            PurePosixPath("my_skill/skill.md"),
            PurePosixPath("my_skill/resource.txt"),
        ]

        root = SkillParser.find_skill_root(paths)

        assert root == PurePosixPath("my_skill")

    def test_find_skill_root_from_archive_root(self):
        paths = [
            PurePosixPath("skill.md"),
            PurePosixPath("assets/readme.txt"),
        ]

        root = SkillParser.find_skill_root(paths)

        assert root == PurePosixPath(".")

    def test_find_skill_root_not_found(self):
        paths = [PurePosixPath("a/b.txt"), PurePosixPath("root.txt")]

        root = SkillParser.find_skill_root(paths)

        assert root is None

    def test_find_skill_md_found_at_root(self):
        paths = [
            PurePosixPath("my_skill/skill.md"),
            PurePosixPath("my_skill/assets/guide.md"),
            PurePosixPath("other/skill.md"),
        ]

        skill_md = SkillParser.find_skill_md(paths, PurePosixPath("my_skill"))

        assert skill_md == PurePosixPath("my_skill/skill.md")

    def test_find_skill_md_not_found(self):
        paths = [PurePosixPath("my_skill/assets/guide.md")]

        skill_md = SkillParser.find_skill_md(paths, PurePosixPath("my_skill"))

        assert skill_md is None

    def test_parse_skill_md_maps_required_and_optional_fields(self):
        text = """---
name: Demo Skill
description: Parse demo
license: 123
compatibility: 3.14
allowed-tools: [search, browse]
metadata:
  owner: dais
---
Body line 1.
Body line 2.
"""

        parsed = SkillParser.parse_skill_md(text)

        assert isinstance(parsed, SkillMd)
        assert parsed.name == "Demo Skill"
        assert parsed.description == "Parse demo"
        assert parsed.license == "123"
        assert parsed.compatibility == "3.14"
        assert parsed.allowed_tools == "['search', 'browse']"
        assert parsed.metadata == {"owner": "dais"}
        assert "Body line 1." in parsed.content

    def test_parse_skill_md_defaults_metadata_to_empty_dict(self):
        text = """---
name: Demo Skill
description: Parse demo
---
Only body.
"""

        parsed = SkillParser.parse_skill_md(text)

        assert parsed.metadata == {}


class TestSkillResource:
    def test_from_bytes_text(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr("dais_sdk.skill.is_binary_string", lambda _: False)

        resource = SkillResource.from_bytes(PurePosixPath("docs/readme.txt"), b"\xef\xbb\xbfhello")

        assert resource.relative == "docs/readme.txt"
        assert resource.type == "text"
        assert resource.content == "hello"

    def test_from_bytes_binary(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr("dais_sdk.skill.is_binary_string", lambda _: True)

        payload = b"\x00\x01\xff\x10"
        resource = SkillResource.from_bytes(PurePosixPath("assets/logo.bin"), payload)

        assert resource.relative == "assets/logo.bin"
        assert resource.type == "binary"
        assert resource.content == payload


class TestSkillFromZip:
    def test_from_zip_success_parses_skill_and_resources(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr("dais_sdk.skill.is_binary_string", lambda data: b"\x00" in data)

        zip_bytes = _make_zip_bytes(
            {
                "my_skill/skill.md": """---
name: Archive Skill
description: Skill from zip
metadata:
  owner: test
---
Main content
""",
                "my_skill/readme.txt": "top-level file should be included in resources",
                "my_skill/assets/guide.txt": "hello resource",
                "my_skill/assets/logo.bin": b"\x00\x01\x02",
            }
        )

        with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as zf:
            skill = Skill.from_zip(zf)

        assert skill.name == "Archive Skill"
        assert skill.description == "Skill from zip"
        assert skill.metadata == {"owner": "test"}

        assert len(skill.resources) == 3
        resources_by_relative = {r.relative: r for r in skill.resources}
        assert set(resources_by_relative) == {"readme.txt", "assets/guide.txt", "assets/logo.bin"}

        readme = resources_by_relative["readme.txt"]
        guide = resources_by_relative["assets/guide.txt"]
        logo = resources_by_relative["assets/logo.bin"]

        assert readme.type == "text"
        assert readme.content == "top-level file should be included in resources"

        assert guide.type == "text"
        assert guide.content == "hello resource"

        assert logo.type == "binary"
        assert logo.content == b"\x00\x01\x02"

    def test_from_zip_raises_when_skill_root_missing(self):
        zip_bytes = _make_zip_bytes({"notes/readme.txt": "no skill markdown"})

        with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as zf:
            with pytest.raises(InvalidSkillArchiveError, match="Skill root not found"):
                Skill.from_zip(zf)

    def test_from_zip_raises_when_skill_md_missing(self, monkeypatch: pytest.MonkeyPatch):
        zip_bytes = _make_zip_bytes(
            {
                "my_skill/skill.md": "---\nname: X\ndescription: Y\n---\nbody",
                "my_skill/assets/file.txt": "data",
            }
        )

        monkeypatch.setattr(SkillParser, "find_skill_md", lambda paths, root: None)

        with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as zf:
            with pytest.raises(InvalidSkillArchiveError, match="SKILL.md not found"):
                Skill.from_zip(zf)
