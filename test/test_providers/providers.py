from dataclasses import dataclass
from pathlib import Path

from fusebuild import Provider


@dataclass
class TestProvider(Provider):
    somefile: str

    def get_some_file(self) -> Path:
        return self.output_dir / self.somefile
