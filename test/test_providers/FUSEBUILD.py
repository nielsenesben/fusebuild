from fusebuild import shell_action, get_action
from test.test_providers.providers import TestProvider
from pathlib import Path

print(f"Loading {__file__}")

shell_action(
    name="hasprovider",
    cmd="echo hallo > $OUTPUT_DIR/somefile",
    providers={"testprovider": TestProvider("somefile")},
)

provider = get_action(Path("."), "hasprovider").providers["testprovider"]

shell_action(
    name="useprovider",
    cmd=f"test hallo = $(cat {provider.get_some_file().absolute()})",
    category="test",
)
