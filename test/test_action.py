import json

import marshmallow_dataclass2
from absl.testing.absltest import TestCase, main  # type: ignore

from fusebuild.core.action import Action, PatternRemapToOutput


class TestExample1(TestCase):
    def test_serialize_no_mappings(self) -> None:
        action = Action(cmd=["true"], category="test")
        schema = marshmallow_dataclass2.class_schema(Action)()
        serialized = json.dumps(schema.dump(action))
        d = json.loads(serialized)
        action_back = schema.load(d)
        self.assertEqual(action_back, action)

    def test_serialize_one_mapping(self) -> None:
        action = Action(
            cmd=["true"],
            category="test",
            mappings=[PatternRemapToOutput("foobar", "barfoo")],
        )
        print(action)
        schema = marshmallow_dataclass2.class_schema(Action)()
        serialized = json.dumps(schema.dump(action))
        print(serialized)
        d = json.loads(serialized)
        action_back = schema.load(d)
        print(action_back)
        self.assertEqual(action_back, action)


if __name__ == "__main__":
    main()
