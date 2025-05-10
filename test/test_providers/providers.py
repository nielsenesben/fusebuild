from fusebuild import Provider
from dataclasses import dataclass


@dataclass
class TestProvider(Provider):
    somefile: str

    def get_some_file(self):
        return self.output_dir / self.somefile
