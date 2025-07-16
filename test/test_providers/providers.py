from dataclasses import dataclass

from fusebuild import Provider


@dataclass
class TestProvider(Provider):
    somefile: str

    def get_some_file(self):
        return self.output_dir / self.somefile
